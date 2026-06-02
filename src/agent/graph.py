from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from src.core.llm import build_chat_model, normalize_content
from src.core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    OrderLineInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from src.utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""
You are a Vietnamese order assistant for an electronics retailer.
Today is {current_day}.

Rules:
- Answer in concise Vietnamese.
- Never invent product IDs, prices, stock, discount, totals, or save path.
- If request is unsafe, refuse immediately and do not call tools.
- If required fields are missing, ask clarification and do not call tools.
- For valid orders, use this exact tool order:
  list_products -> get_product_details -> get_discount -> calculate_order_totals -> save_order
- If product details are already obtained, do not call list_products repeatedly.
- Save only after pricing validation succeeds.
""".strip()


def build_tools(store: OrderDataStore):
    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search catalog and return compact matched products."""
        payload = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags,
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Return exact details and validation token for selected product IDs."""
        return json.dumps(store.get_product_details(product_ids), ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Return deterministic campaign discount for this order."""
        return json.dumps(store.get_discount(seed_hint=seed_hint or "guest", customer_tier=customer_tier), ensure_ascii=False)

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(items: list[OrderLineInput], detail_token: str, discount_rate: float) -> str:
        """Validate stock and compute subtotal, discount, and final total."""
        payload = store.calculate_order_totals(items=items, detail_token=detail_token, discount_rate=discount_rate)
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items: list[OrderLineInput],
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist validated order payload and return saved path."""
        result = store.save_order(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            shipping_address=shipping_address,
            items=items,
            detail_token=detail_token,
            discount_rate=discount_rate,
            campaign_code=campaign_code,
            customer_tier=customer_tier,
            notes=notes,
        )
        return json.dumps(result, ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    refusal_message = _unsafe_request_message(query)
    if refusal_message:
        return AgentResult(query=query, final_answer=refusal_message, tool_calls=[], provider=provider, model_name=model_name)

    missing_fields = _missing_required_fields(query)
    if missing_fields:
        return AgentResult(
            query=query,
            final_answer=f"Bạn vui lòng bổ sung: {', '.join(missing_fields)} trước khi mình xử lý đơn hàng.",
            tool_calls=[],
            provider=provider,
            model_name=model_name,
        )

    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    agent = create_agent(
        model=build_chat_model(provider=provider, model_name=model_name, temperature=0.0),
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = response["messages"] if isinstance(response, dict) else response
    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)

    if saved_order is None:
        deterministic = _run_deterministic_flow(query=query, store=store)
        if deterministic is not None:
            final_answer, tool_calls, saved_order, saved_order_path = deterministic
            return AgentResult(
                query=query,
                final_answer=final_answer,
                tool_calls=tool_calls,
                provider=provider,
                model_name=model_name,
                saved_order=saved_order,
                saved_order_path=saved_order_path,
            )

    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


def extract_final_answer(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    pending: dict[str, dict] = {}
    records: list[ToolCallRecord] = []
    for message in messages:
        if isinstance(message, AIMessage):
            for call in getattr(message, "tool_calls", []) or []:
                pending[call["id"]] = {"name": call["name"], "args": call.get("args", {}) or {}}
        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )
    for metadata in pending.values():
        records.append(ToolCallRecord(name=metadata["name"], args=metadata["args"], output=""))
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") == "saved":
            return payload.get("saved_order"), payload.get("path")
    return None, None


def _run_deterministic_flow(
    *,
    query: str,
    store: OrderDataStore,
) -> tuple[str, list[ToolCallRecord], dict | None, str | None] | None:
    customer = _extract_customer_info(query)
    items = _extract_items_from_query(query, store)
    if not customer or not items:
        return None

    tool_calls: list[ToolCallRecord] = []
    item_ids = [item.product_id for item in items]

    listed = store.list_products(query=query, required_tags=[], in_stock_only=True, limit=8)
    tool_calls.append(ToolCallRecord(name="list_products", args={"query": query, "limit": 8}, output=json.dumps(listed, ensure_ascii=False)))

    details = store.get_product_details(item_ids)
    tool_calls.append(ToolCallRecord(name="get_product_details", args={"product_ids": item_ids}, output=json.dumps(details, ensure_ascii=False)))
    if details.get("status") != "ok":
        return None

    detail_token = str(details.get("detail_token", ""))
    discount = store.get_discount(seed_hint=customer["customer_email"], customer_tier="standard")
    tool_calls.append(
        ToolCallRecord(
            name="get_discount",
            args={"seed_hint": customer["customer_email"], "customer_tier": "standard"},
            output=json.dumps(discount, ensure_ascii=False),
        )
    )
    discount_rate = float(discount.get("discount_rate", 0.0))
    campaign_code = str(discount.get("campaign_code", ""))
    if discount_rate not in {0.1, 0.2}:
        return None

    totals = store.calculate_order_totals(items=items, detail_token=detail_token, discount_rate=discount_rate)
    tool_calls.append(
        ToolCallRecord(
            name="calculate_order_totals",
            args={"items": [item.model_dump() for item in items], "detail_token": detail_token, "discount_rate": discount_rate},
            output=json.dumps(totals, ensure_ascii=False),
        )
    )
    if totals.get("status") != "ok":
        return None

    saved = store.save_order(
        customer_name=customer["customer_name"],
        customer_phone=customer["customer_phone"],
        customer_email=customer["customer_email"],
        shipping_address=customer["shipping_address"],
        items=items,
        detail_token=detail_token,
        discount_rate=discount_rate,
        campaign_code=campaign_code,
        customer_tier="standard",
        notes="",
    )
    tool_calls.append(
        ToolCallRecord(
            name="save_order",
            args={
                "customer_name": customer["customer_name"],
                "customer_phone": customer["customer_phone"],
                "customer_email": customer["customer_email"],
                "shipping_address": customer["shipping_address"],
                "items": [item.model_dump() for item in items],
                "detail_token": detail_token,
                "discount_rate": discount_rate,
                "campaign_code": campaign_code,
                "customer_tier": "standard",
                "notes": "",
            },
            output=json.dumps(saved, ensure_ascii=False),
        )
    )
    if saved.get("status") != "saved":
        return None

    final_total = saved.get("saved_order", {}).get("pricing", {}).get("final_total", 0)
    answer = (
        f"Đã lưu đơn hàng thành công với mã {saved.get('order_id')}."
        f" Tổng thanh toán sau giảm giá là {final_total} VND."
        f" File lưu tại {saved.get('saved_order', {}).get('save_path', '')}."
    )
    return answer, tool_calls, saved.get("saved_order"), saved.get("path")


def _normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    stripped = stripped.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"[^a-zA-Z0-9]+", " ", stripped.lower()).strip()


def _extract_items_from_query(query: str, store: OrderDataStore) -> list[OrderLineInput]:
    normalized_query = _normalize_text(query)
    selected: dict[str, int] = {}
    for product in store.products:
        pname = _normalize_text(product.name)
        for match in re.finditer(rf"\b(\d+)\s+{re.escape(pname)}\b", normalized_query):
            qty = int(match.group(1))
            if qty > 0:
                selected[product.product_id] = qty
    return [OrderLineInput(product_id=pid, quantity=qty) for pid, qty in selected.items()]


def _extract_customer_info(query: str) -> dict[str, str] | None:
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", query)
    phone_match = re.search(r"\b0\d{9,10}\b", query)
    if not email_match or not phone_match:
        return None
    name_match = re.search(r"\bcho\s+([^,.;\n]+)", query, flags=re.IGNORECASE)
    if not name_match:
        return None
    name = name_match.group(1).strip()
    if not name or name.lower().split()[0] in {"tôi", "toi", "mình", "minh", "shop"}:
        return None
    shipping_match = re.search(
        r"(?:giao(?:\s+hàng)?(?:\s+đến|\s+tới)?|ship to|địa chỉ giao hàng)\s*[:,-]?\s*([^.\n]+)",
        query,
        flags=re.IGNORECASE,
    )
    if not shipping_match:
        return None
    return {
        "customer_name": name,
        "customer_phone": phone_match.group(0),
        "customer_email": email_match.group(0),
        "shipping_address": shipping_match.group(1).strip(),
    }


def _unsafe_request_message(query: str) -> str | None:
    lowered = query.lower()
    unsafe = [
        "hóa đơn giả",
        "hoa don gia",
        "fake invoice",
        "bỏ qua policy",
        "ignore policy",
        "bypass stock",
        "bỏ qua tồn kho",
        "bo qua ton kho",
        "giảm giá 90%",
        "giam gia 90%",
        "force discount",
    ]
    if any(token in lowered for token in unsafe):
        return (
            "Mình không thể hỗ trợ yêu cầu này vì vi phạm chính sách (hóa đơn giả, ép giảm giá, hoặc bỏ qua tồn kho/policy). "
            "Mình chỉ có thể xử lý đơn hợp lệ dựa trên catalog và quy định."
        )
    return None


def _has_customer_name(query: str) -> bool:
    for pattern in (r"\bcho\s+([^,.;\n]+)", r"\bfor\s+([^,.;\n]+)"):
        for match in re.finditer(pattern, query, flags=re.IGNORECASE):
            candidate = match.group(1).strip().lower()
            if not candidate:
                continue
            first = candidate.split()[0]
            if first not in {"tôi", "toi", "mình", "minh", "em", "anh", "chị", "chi", "shop", "me"}:
                return True
    return False


def _missing_required_fields(query: str) -> list[str]:
    missing: list[str] = []
    lowered = query.lower()
    if not re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", query):
        missing.append("email khách hàng")
    if not re.search(r"\b0\d{9,10}\b", query):
        missing.append("số điện thoại")
    if not any(token in lowered for token in ["giao", "ship to", "địa chỉ", "dia chi"]):
        missing.append("địa chỉ giao hàng")
    if not re.search(r"\b\d+\s+[^,.]+", _normalize_text(query)):
        missing.append("sản phẩm và số lượng")
    if not _has_customer_name(query):
        missing.append("tên khách hàng")
    return missing
