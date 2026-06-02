from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

from src.core.schemas import OrderLineInput, ProductRecord


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    compact = re.sub(r"[^a-zA-Z0-9]+", " ", stripped.lower())
    return re.sub(r"\s+", " ", compact).strip()


class OrderDataStore:
    def __init__(self, data_dir: Path, output_dir: Path, *, today: str | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.today = today or "2026-06-01"
        raw_products = json.loads((self.data_dir / "products.json").read_text(encoding="utf-8"))
        self.products = [ProductRecord(**item) for item in raw_products]
        self.product_index = {item.product_id: item for item in self.products}
        self.category_aliases = {
            "laptop": "laptop",
            "notebook": "laptop",
            "monitor": "monitor",
            "screen": "monitor",
            "man hinh": "monitor",
            "mouse": "mouse",
            "chuot": "mouse",
            "keyboard": "keyboard",
            "ban phim": "keyboard",
            "headphone": "headphone",
            "tai nghe": "headphone",
            "dock": "dock",
            "storage": "storage",
            "ssd": "storage",
            "stand": "stand",
            "webcam": "webcam",
        }

    @staticmethod
    def build_detail_token(product_ids: list[str]) -> str:
        normalized = "|".join(sorted(product_ids))
        return "DET-" + hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10].upper()

    def validate_detail_token(self, product_ids: list[str], detail_token: str) -> bool:
        return detail_token == self.build_detail_token(product_ids)

    def canonicalize_category(self, value: str | None) -> str | None:
        if not value:
            return None
        return self.category_aliases.get(_normalize(value), _normalize(value))

    def list_products(
        self,
        *,
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> list[dict]:
        normalized_query = _normalize(query or "")
        query_terms = [term for term in normalized_query.split() if term and not term.isdigit() and len(term) > 1]
        wanted_category = self.canonicalize_category(category)
        wanted_tags = {_normalize(tag) for tag in (required_tags or []) if tag.strip()}
        results: list[tuple[int, int, str, dict]] = []

        for product in self.products:
            if in_stock_only and product.stock <= 0:
                continue
            if wanted_category and product.category != wanted_category:
                continue
            if max_unit_price is not None and product.unit_price > max_unit_price:
                continue

            haystack = _normalize(" ".join([product.name, product.brand, product.category, product.description, *product.tags]))
            score = 0
            matched_terms: list[str] = []
            for term in query_terms:
                if term in haystack:
                    score += 2
                    matched_terms.append(term)
            for tag in wanted_tags:
                if tag in haystack:
                    score += 3
                    matched_terms.append(tag)
                else:
                    score -= 1
            if wanted_category:
                score += 3
            if query_terms and not matched_terms:
                continue
            results.append(
                (
                    score,
                    product.stock,
                    product.product_id,
                    {
                        "product_id": product.product_id,
                        "name": product.name,
                        "brand": product.brand,
                        "category": product.category,
                        "tags": product.tags,
                        "matched_terms": sorted(set(matched_terms)),
                        "next_step": "Call get_product_details with the chosen product_id list to verify price, stock, and the detail_token.",
                    },
                )
            )

        results.sort(key=lambda item: (-item[0], self.product_index[item[2]].unit_price, item[2]))
        return [item[-1] for item in results[:limit]]

    def get_product_details(self, product_ids: list[str]) -> dict:
        details: list[dict] = []
        for product_id in product_ids:
            product = self.product_index.get(product_id)
            if not product:
                details.append({"product_id": product_id, "status": "not_found"})
                continue
            details.append(
                {
                    "status": "ok",
                    "product_id": product.product_id,
                    "sku": product.sku,
                    "name": product.name,
                    "brand": product.brand,
                    "category": product.category,
                    "unit_price": product.unit_price,
                    "stock": product.stock,
                    "warranty_months": product.warranty_months,
                    "tags": product.tags,
                    "description": product.description,
                }
            )
        found_product_ids = [item["product_id"] for item in details if item.get("status") == "ok"]
        return {
            "status": "ok" if found_product_ids else "error",
            "detail_token": self.build_detail_token(found_product_ids) if found_product_ids else "",
            "items": details,
        }

    def get_discount(self, *, seed_hint: str, customer_tier: str = "standard") -> dict:
        normalized_seed = seed_hint.strip().lower()
        digest = hashlib.sha256(f"{customer_tier}|{normalized_seed}".encode("utf-8")).hexdigest()
        discount_rate = 0.2 if int(digest[-2:], 16) % 10 < 4 else 0.1
        return {
            "status": "ok",
            "seed_hint": seed_hint,
            "customer_tier": customer_tier,
            "discount_rate": discount_rate,
            "campaign_code": f"FLASH-{int(discount_rate * 100):02d}",
        }

    def calculate_order_totals(self, *, items: list[OrderLineInput], detail_token: str, discount_rate: float) -> dict:
        if discount_rate not in {0.1, 0.2}:
            return {"status": "error", "errors": [f"Unsupported discount rate: {discount_rate}."]}
        requested_product_ids = [item.product_id for item in items]
        if not self.validate_detail_token(requested_product_ids, detail_token):
            return {
                "status": "error",
                "errors": ["Invalid detail token. Call get_product_details again before pricing this order."],
            }

        errors: list[str] = []
        lines: list[dict] = []
        subtotal = 0
        for item in sorted(items, key=lambda current: current.product_id):
            product = self.product_index.get(item.product_id)
            if not product:
                errors.append(f"Unknown product_id: {item.product_id}.")
                continue
            if item.quantity > product.stock:
                errors.append(f"Insufficient stock for {product.name}: requested {item.quantity}, available {product.stock}.")
                continue
            line_total = product.unit_price * item.quantity
            subtotal += line_total
            lines.append(
                {
                    "product_id": product.product_id,
                    "sku": product.sku,
                    "name": product.name,
                    "category": product.category,
                    "quantity": item.quantity,
                    "unit_price": product.unit_price,
                    "line_total": line_total,
                }
            )

        if errors:
            return {"status": "error", "errors": errors, "items": lines}

        discount_amount = int(subtotal * discount_rate)
        final_total = subtotal - discount_amount
        return {
            "status": "ok",
            "items": lines,
            "pricing": {
                "currency": "VND",
                "subtotal": subtotal,
                "discount_rate": discount_rate,
                "discount_amount": discount_amount,
                "final_total": final_total,
            },
            "detail_token": detail_token,
        }

    def save_order(
        self,
        *,
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
    ) -> dict:
        pricing_snapshot = self.calculate_order_totals(items=items, detail_token=detail_token, discount_rate=discount_rate)
        if pricing_snapshot["status"] != "ok":
            return pricing_snapshot

        normalized_items = sorted(
            [{"product_id": item.product_id, "quantity": item.quantity} for item in items],
            key=lambda current: current["product_id"],
        )
        seed_payload = json.dumps(
            {
                "customer_email": customer_email.strip().lower(),
                "customer_phone": "".join(ch for ch in customer_phone if ch.isdigit()),
                "items": normalized_items,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        order_id = "ORD-" + hashlib.sha1(seed_payload.encode("utf-8")).hexdigest()[:10].upper()
        relative_path = Path("artifacts") / "orders" / f"{order_id}.json"
        absolute_path = self.output_dir / f"{order_id}.json"

        payload = {
            "order_id": order_id,
            "created_at": self.today,
            "status": "confirmed",
            "customer": {
                "name": customer_name.strip(),
                "phone": customer_phone.strip(),
                "email": customer_email.strip(),
                "shipping_address": shipping_address.strip(),
            },
            "items": pricing_snapshot["items"],
            "pricing": pricing_snapshot["pricing"],
            "discount": {
                "campaign_code": campaign_code,
                "customer_tier": customer_tier,
            },
            "notes": notes.strip(),
            "save_path": relative_path.as_posix(),
            "source": "llm-order-agent",
        }
        absolute_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "status": "saved",
            "order_id": order_id,
            "path": str(absolute_path),
            "saved_order": payload,
        }
