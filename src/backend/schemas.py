"""
DeepRetail — Pydantic Schemas for FastAPI request/response models.
"""

from typing import Optional
from pydantic import BaseModel


class CartItem(BaseModel):
    name: str
    quantity: int
    unit_price: float
    subtotal: float


class CartResponse(BaseModel):
    session_id: str
    items: list[CartItem]
    total: float


class ProductInfo(BaseModel):
    product_id: int
    name: str
    category: str
    price: float
    stock: int
    shelf_slot: Optional[str]
    low_stock_threshold: int


class InventoryResponse(BaseModel):
    products: list[ProductInfo]
    low_stock: list[ProductInfo]


class AlertItem(BaseModel):
    type: str           # "LOW_STOCK" | "MISPLACED" | "ANOMALY" | "UNKNOWN_PRODUCT"
    message: str
    timestamp: float
    details: dict = {}


class AlertsResponse(BaseModel):
    alerts: list[AlertItem]


class EventItem(BaseModel):
    timestamp: float
    event: str          # "pick" | "return" | "anomaly"
    track_id: int
    product: str
    conf: float
    session_id: str


class TopProductItem(BaseModel):
    product_id: int
    name: str
    pick_count: int


class RecommendationItem(BaseModel):
    product_id: int
    name: str
    co_occurrence_count: int
    confidence: float


class AnalyticsTopResponse(BaseModel):
    top_products: list[TopProductItem]


class AnalyticsRecsResponse(BaseModel):
    product_id: int
    product_name: str
    recommendations: list[RecommendationItem]


class PipelineStatusResponse(BaseModel):
    running: bool
    fps: float
    frame_id: int
    active_tracks: int
    session_id: str


class ForecastItem(BaseModel):
    product_id: int
    name: str
    stock: int
    history_days: int
    method: str                     # "ses" | "mean" (short history)
    alpha: Optional[float]
    forecast: float                 # units / day
    sigma: float
    mae_ses: Optional[float]
    mae_naive: Optional[float]
    mae_ma: Optional[float]
    safety_stock: float
    reorder_point: float
    order_qty: int
    days_to_stockout: Optional[float]
    reorder_now: bool


class ForecastResponse(BaseModel):
    products: list[ForecastItem]
