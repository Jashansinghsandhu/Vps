"""
FINAL CONSOLIDATED TELEGRAM CRYPTO GAMBLING BOT (Parts 1-5 Integrated)
======================================================================

IMPORTANT: This single file merges all previously provided parts (1–5). 
User mention "1 to 6": Part 6 was never delivered; integration covers through Part 5. 
If you later need a Part 6 feature set (advanced tournament logic etc.), request explicitly.

BEFORE RUNNING:
1. Python 3.11+ recommended.
2. Install dependencies (minimum):
   pip install python-telegram-bot~=21.4 SQLAlchemy~=2.0 psycopg2-binary aiosqlite web3~=6.15 cryptography fastapi uvicorn
   (Optional extras for performance: uvloop, hypothesis for property tests)
3. Set environment variables (export or .env):
   BOT_TOKEN=YOUR_TELEGRAM_BOT_TOKEN
   DB_URL=postgresql+psycopg2://user:pass@host/db   (or sqlite:///casino.db for dev)
   ADMIN_IDS='[123456789,987654321]'   (Telegram numeric user IDs as JSON array)
   CALLBACK_TOKEN_SECRET=CHANGE_ME_SECRET
   SESSION_TOKEN_SECRET=CHANGE_ME_SESSION_SECRET
   ETH_RPC_URL=...
   BNB_RPC_URL=...
   SUPPORTED_CHAINS='["ETH","BNB"]'
   (Optional) BLOCKED_REGIONS="US,CA"   (example)
4. SECURITY / COMPLIANCE:
   - This is a prototype. Real-money deployment requires licensing, AML/KYC workflows, age checks, geofencing.
   - Private keys / custody NOT implemented here (only placeholders).
   - Review all code paths for race conditions, accounting correctness, jurisdictional compliance.
5. RUN:
   python final.py
   The bot starts polling Telegram and the FastAPI server (default port 8000).

FEATURE SUMMARY (Integrated):
- Provably Fair (HMAC) seeds, rotation & verification
- Games: dice, limbo, coinflip, roulette (partial), blackjack_full (interactive), mines (static + interactive),
         slots (weighted + jackpot), tower, keno, crash (real-time), poker_house, holdem_house (PvP), pvp_dice_emoji, darts_pvp
- VIP, XP, rewards (weekly, monthly, rebate)
- Tipping, rain, airdrop placeholder hooks
- Admin panel: edges, limits, freeze/unfreeze, seed rotation, withdrawals, exposure snapshots, collusion, jackpot, KYC flags
- PvP matches (emoji dice / darts)
- Crash real-time betting & websocket feed
- Progressive slot jackpot
- Multi-seat Hold'em simplified engine
- RTP report & analytics (audit logs, exposure snapshots)
- Security: signed callback data, session tokens for REST API, replay protection, flood control
- REST API + WebSocket endpoints (/api/..., /ws/...)
- Property/self tests & Monte Carlo EV simulations

DISCLAIMER:
This code is large and for educational use. Expect to refactor into modules for maintainability.

"""

from __future__ import annotations
import asyncio
import hmac
import hashlib
import json
import logging
import os
import random
import secrets
import string
import time
import base64
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, getcontext, ROUND_DOWN
from typing import Any, Dict, List, Optional, Tuple, Sequence

# Third-party imports
from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup,
    KeyboardButton, Message, User as TgUser, BotCommand
)
from telegram.ext import (
    ApplicationBuilder, Application, CommandHandler, CallbackQueryHandler,
    ContextTypes, MessageHandler, filters, AIORateLimiter
)

from sqlalchemy import (
    create_engine, String, Integer, DateTime, Numeric, ForeignKey, Boolean, Text, JSON as SA_JSON,
    select, func, update as sa_update
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from sqlalchemy.exc import SQLAlchemyError

try:
    from web3 import Web3, HTTPProvider
except ImportError:
    Web3 = None

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse
except ImportError:
    FastAPI = None

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

class Config:
    BOT_TOKEN: str = os.getenv("BOT_TOKEN", "8396606980:AAE3-QlN2wRq83jNgqKoUqM8v6PgrMUTR98")
    DB_URL: str = os.getenv("DB_URL", "sqlite:///casino.db")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ENV: str = os.getenv("ENV", "dev")

    SUPPORTED_CHAINS: List[str] = json.loads(os.getenv("SUPPORTED_CHAINS", '["ETH","BNB"]'))
    DEFAULT_CONFIRMATIONS: int = int(os.getenv("DEFAULT_CONFIRMATIONS", "12"))

    HOUSE_EDGES: Dict[str, float] = {
        "dice": 0.01, "limbo": 0.01, "coinflip": 0.01, "roulette": 0.027,
        "blackjack": 0.006, "slots": 0.05, "mines": 0.02, "tower": 0.02,
        "keno": 0.25, "crash": 0.015, "pvp_rake": 0.03
    }

    MIN_BET: float = float(os.getenv("MIN_BET", "0.0001"))
    MAX_BET: float = float(os.getenv("MAX_BET", "10"))
    MAX_WIN: float = float(os.getenv("MAX_WIN", "1000"))
    SEED_ROTATION_BETS: int = int(os.getenv("SEED_ROTATION_BETS", "5000"))

    XP_RATE: float = 1.0
    VIP_TIERS: List[Tuple[str, float]] = [
        ("Bronze", 0), ("Silver", 100), ("Gold", 1000), ("Platinum", 10000), ("Diamond", 50000)
    ]
    WEEKLY_REWARD_RATE: float = 0.002
    MONTHLY_REWARD_RATE: float = 0.005
    REBATE_RATE_BASE: float = 0.01
    REWARD_EXPIRE_DAYS: int = 14

    CALLBK_SECRET: str = os.getenv("CALLBACK_TOKEN_SECRET", "dev_secret_change_me")
    SESSION_TOKEN_SECRET: str = os.getenv("SESSION_TOKEN_SECRET", "dev_session_secret_change")
    FLOOD_WINDOW_SEC: int = 10
    FLOOD_MAX_ACTIONS: int = 25

    ETH_RPC_URL: str = os.getenv("ETH_RPC_URL", "https://mainnet.infura.io/v3/25cdeb5b655744f2b6d88c998e55eace")
    BNB_RPC_URL: str = os.getenv("BNB_RPC_URL", "https://bsc-dataseed.binance.org/")
    CHAIN_POLL_INTERVAL: int = int(os.getenv("CHAIN_POLL_INTERVAL", "30"))

    SELF_EXCLUSION_DAYS_MIN: int = 1
    SELF_EXCLUSION_DAYS_MAX: int = 365

    # Crash settings
    CRASH_GROWTH_RATE: float = 0.04
    CRASH_MAX_MULT: float = 500

    # Jackpot
    JACKPOT_CONTRIB_RATE: float = 0.01
    JACKPOT_HIT_CHANCE: float = 1 / 5000

getcontext().prec = 28

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("casino_bot")

# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def hmac_sha256(key: str, msg: str) -> str:
    return hmac.new(key.encode(), msg.encode(), hashlib.sha256).hexdigest()

def random_seed(length: int = 64) -> str:
    return secrets.token_hex(length // 2)

def hash_seed(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()

def safe_floor_decimal(amount: Decimal, decimals: int = 8) -> Decimal:
    quant = Decimal("1." + "0"*decimals)
    return amount.quantize(quant, rounding=ROUND_DOWN)

def format_amount(v) -> str:
    return f"{Decimal(v):f}"

def compute_vip_tier(lifetime_wager: float) -> str:
    tier = Config.VIP_TIERS[0][0]
    for name, threshold in Config.VIP_TIERS:
        if lifetime_wager >= threshold:
            tier = name
        else:
            break
    return tier

def clamp_bet(amount: Decimal) -> Decimal:
    if amount < Decimal(str(Config.MIN_BET)):
        raise ValueError("Below min bet")
    if amount > Decimal(str(Config.MAX_BET)):
        raise ValueError("Above max bet")
    return amount

def responsible_gambling_notice() -> str:
    return "Responsible Gambling: Use /selfexclude or /cooldown for breaks."

def generate_session_token(tg_id: int, ttl_minutes: int = 60) -> str:
    payload = {
        "tg": tg_id,
        "exp": int(time.time()) + ttl_minutes * 60,
        "nonce": random_seed(16)
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    sig = hmac.new(Config.SESSION_TOKEN_SECRET.encode(), body.encode(), hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(body.encode()).decode() + "." + sig

def verify_session_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        body_b64, sig = token.split(".", 1)
        body = base64.urlsafe_b64decode(body_b64.encode()).decode()
        expected = hmac.new(Config.SESSION_TOKEN_SECRET.encode(), body.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, sig):
            return None
        payload = json.loads(body)
        if payload["exp"] < time.time():
            return None
        return payload
    except Exception:
        return None

# Flood control
USER_ACTION_LOG: Dict[int, List[float]] = {}
async def flood_control(update: Update):
    if not update.effective_user:
        return
    uid = update.effective_user.id
    lst = USER_ACTION_LOG.setdefault(uid, [])
    now = time.time()
    lst.append(now)
    window_start = now - Config.FLOOD_WINDOW_SEC
    while lst and lst[0] < window_start:
        lst.pop(0)
    if len(lst) > Config.FLOOD_MAX_ACTIONS:
        if update.message:
            try: await update.message.reply_text("Rate limited.")
            except: pass
        raise Exception("Flood limit")

def is_valid_evm_address(addr: str) -> bool:
    if not addr.startswith("0x") or len(addr) != 42:
        return False
    try: int(addr[2:], 16)
    except: return False
    return True

def is_admin(user_id: int) -> bool:
    try:
        admin_ids = json.loads(os.getenv("ADMIN_IDS", "[6083286836]"))
        return user_id in admin_ids
    except:
        return False

# ---------------------------------------------------------------------
# DATABASE MODELS
# ---------------------------------------------------------------------

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tg_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    username: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    balance: Mapped[Decimal] = mapped_column(Numeric(36,18), default=Decimal("0"))
    vip_tier: Mapped[str] = mapped_column(String(32), default="Bronze")
    xp: Mapped[int] = mapped_column(Integer, default=0)
    lifetime_wager: Mapped[Decimal] = mapped_column(Numeric(36,18), default=Decimal("0"))
    client_seed: Mapped[str] = mapped_column(String(128), default=random_seed)
    current_server_seed_id: Mapped[Optional[int]] = mapped_column(ForeignKey("seeds.id"))
    server_seed_hash: Mapped[Optional[str]] = mapped_column(String(128))
    bets_count_in_seed: Mapped[int] = mapped_column(Integer, default=0)
    self_excluded_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    kyc_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    frozen: Mapped[bool] = mapped_column(Boolean, default=False)

class ServerSeed(Base):
    __tablename__ = "seeds"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    server_seed: Mapped[str] = mapped_column(String(128))
    server_seed_hash: Mapped[str] = mapped_column(String(128), index=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    rotated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

class Bet(Base):
    __tablename__ = "bets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    game: Mapped[str] = mapped_column(String(32), index=True)
    amount: Mapped[Decimal] = mapped_column(Numeric(36,18))
    result: Mapped[str] = mapped_column(String(64))
    payout: Mapped[Decimal] = mapped_column(Numeric(36,18))
    nonce: Mapped[int] = mapped_column(Integer)
    server_seed_id: Mapped[int] = mapped_column(ForeignKey("seeds.id"))
    client_seed_snapshot: Mapped[str] = mapped_column(String(128))
    outcome_data: Mapped[Dict[str, Any]] = mapped_column(SA_JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

class DepositTx(Base):
    __tablename__ = "tx_deposits"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    chain: Mapped[str] = mapped_column(String(16))
    address: Mapped[str] = mapped_column(String(128))
    tx_hash: Mapped[str] = mapped_column(String(128), unique=True)
    amount: Mapped[Decimal] = mapped_column(Numeric(36,18))
    confirmations: Mapped[int] = mapped_column(Integer, default=0)
    credited: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

class WithdrawalTx(Base):
    __tablename__ = "tx_withdrawals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    chain: Mapped[str] = mapped_column(String(16))
    address: Mapped[str] = mapped_column(String(128))
    amount: Mapped[Decimal] = mapped_column(Numeric(36,18))
    status: Mapped[str] = mapped_column(String(16), default="pending")
    admin_id: Mapped[Optional[int]] = mapped_column(Integer)
    tx_hash: Mapped[Optional[str]] = mapped_column(String(128))
    reason: Mapped[Optional[str]] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

class Reward(Base):
    __tablename__ = "rewards"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    type: Mapped[str] = mapped_column(String(16))
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    amount: Mapped[Decimal] = mapped_column(Numeric(36,18))
    claimed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

class Tip(Base):
    __tablename__ = "tips"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    from_user: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    to_user: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    amount: Mapped[Decimal] = mapped_column(Numeric(36,18))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

class Rain(Base):
    __tablename__ = "rains"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    from_user: Mapped[int] = mapped_column(ForeignKey("users.id"))
    total_amount: Mapped[Decimal] = mapped_column(Numeric(36,18))
    recipients_count: Mapped[int] = mapped_column(Integer)
    completed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

class Leaderboard(Base):
    __tablename__ = "leaderboards"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    period: Mapped[str] = mapped_column(String(16))
    metric: Mapped[str] = mapped_column(String(32))
    snapshot_json: Mapped[Dict[str, Any]] = mapped_column(SA_JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

class PvPMatch(Base):
    __tablename__ = "pvp_matches"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game: Mapped[str] = mapped_column(String(32), index=True)
    creator_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    opponent_user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"))
    entry_fee: Mapped[Decimal] = mapped_column(Numeric(36,18))
    rake: Mapped[Decimal] = mapped_column(Numeric(36,18), default=Decimal("0"))
    pot: Mapped[Decimal] = mapped_column(Numeric(36,18), default=Decimal("0"))
    best_of: Mapped[int] = mapped_column(Integer, default=1)
    state_json: Mapped[Dict[str, Any]] = mapped_column(SA_JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="waiting")
    winner_user_id: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

class SlotJackpot(Base):
    __tablename__ = "slot_jackpots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pool: Mapped[Decimal] = mapped_column(Numeric(36,18), default=Decimal("0"))
    last_hit_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

class GameSession(Base):
    __tablename__ = "game_sessions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    game: Mapped[str] = mapped_column(String(32), index=True)
    state_json: Mapped[Dict[str, Any]] = mapped_column(SA_JSON, default=dict)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    category: Mapped[str] = mapped_column(String(32), index=True)
    actor_user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"))
    target_user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"))
    reference: Mapped[Optional[str]] = mapped_column(String(64))
    data: Mapped[Dict[str, Any]] = mapped_column(SA_JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

class ExposureSnapshot(Base):
    __tablename__ = "exposure_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    total_user_balances: Mapped[Decimal] = mapped_column(Numeric(36,18))
    jackpot_pool: Mapped[Decimal] = mapped_column(Numeric(36,18))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    notes: Mapped[Optional[str]] = mapped_column(String(128))

class UserDeviceFingerprint(Base):
    __tablename__ = "user_device_fingerprints"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    fingerprint_hash: Mapped[str] = mapped_column(String(128), index=True)
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    usage_count: Mapped[int] = mapped_column(Integer, default=1)

class HoldemTable(Base):
    __tablename__ = "holdem_tables"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    big_blind: Mapped[Decimal] = mapped_column(Numeric(36,18))
    max_players: Mapped[int] = mapped_column(Integer)
    rake_rate: Mapped[Decimal] = mapped_column(Numeric(10,6), default=Decimal("0.03"))
    status: Mapped[str] = mapped_column(String(16), default="waiting")
    state_json: Mapped[Dict[str, Any]] = mapped_column(SA_JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

class HoldemSeat(Base):
    __tablename__ = "holdem_seats"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    table_id: Mapped[int] = mapped_column(ForeignKey("holdem_tables.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    seat_index: Mapped[int] = mapped_column(Integer)
    stack: Mapped[Decimal] = mapped_column(Numeric(36,18))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    all_in: Mapped[bool] = mapped_column(Boolean, default=False)
    joined_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

# ---------------------------------------------------------------------
# DB INIT
# ---------------------------------------------------------------------

engine = create_engine(Config.DB_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

def init_db():
    Base.metadata.create_all(engine)

# ---------------------------------------------------------------------
# PROVABLY FAIR
# ---------------------------------------------------------------------

class ProvablyFairEngine:
    @staticmethod
    def hmac_roll(server_seed: str, client_seed: str, nonce: int) -> str:
        return hmac_sha256(server_seed, f"{client_seed}:{nonce}")

    @staticmethod
    def digest_to_uniform(hex_digest: str, offset: int = 0) -> float:
        start = offset * 16
        slice_hex = hex_digest[start:start+16]
        if len(slice_hex) < 16:
            slice_hex = hex_digest[-16:]
        val = int(slice_hex, 16)
        return val / 0xFFFFFFFFFFFF0000  # approximate 2^64

    @staticmethod
    def dice_roll(server_seed: str, client_seed: str, nonce: int) -> float:
        h = ProvablyFairEngine.hmac_roll(server_seed, client_seed, nonce)
        u = ProvablyFairEngine.digest_to_uniform(h)
        return u * 100  # 0 - 100

    @staticmethod
    def limbo_point(server_seed: str, client_seed: str, nonce: int) -> float:
        h = ProvablyFairEngine.hmac_roll(server_seed, client_seed, nonce)
        u = ProvablyFairEngine.digest_to_uniform(h)
        epsilon = 1e-12
        raw = 1 / max(epsilon, (1 - u))
        return min(raw, 100000)

    @staticmethod
    def coin_flip(server_seed: str, client_seed: str, nonce: int) -> str:
        h = ProvablyFairEngine.hmac_roll(server_seed, client_seed, nonce)
        return "heads" if ProvablyFairEngine.digest_to_uniform(h) < 0.5 else "tails"

    @staticmethod
    def roulette_number(server_seed: str, client_seed: str, nonce: int) -> int:
        h = ProvablyFairEngine.hmac_roll(server_seed, client_seed, nonce)
        return int(ProvablyFairEngine.digest_to_uniform(h) * 37)

    @staticmethod
    def crash_multiplier(server_seed: str, client_seed: str, nonce: int) -> float:
        return ProvablyFairEngine.limbo_point(server_seed, client_seed, nonce)

    @staticmethod
    def mines_positions(server_seed: str, client_seed: str, nonce: int, grid_size: int, mines: int) -> List[int]:
        h = ProvablyFairEngine.hmac_roll(server_seed, client_seed, nonce)
        positions = list(range(grid_size))
        for i in range(grid_size - 1, 0, -1):
            idx_uniform = ProvablyFairEngine.digest_to_uniform(h, offset=(grid_size - i) % 4)
            j = int(idx_uniform * (i + 1))
            positions[i], positions[j] = positions[j], positions[i]
        return positions[:mines]

class ExtendedPF:
    @staticmethod
    def weighted_index(hmac_hex: str, offset: int, weights: List[int]) -> int:
        total = sum(weights)
        segment = hmac_hex[offset*8:(offset+1)*8]
        if len(segment) < 8: segment = hmac_hex[-8:]
        r_int = int(segment, 16)
        pick = r_int % total
        cum = 0
        for i, w in enumerate(weights):
            cum += w
            if pick < cum:
                return i
        return len(weights)-1

    @staticmethod
    def slots_reels(server_seed: str, client_seed: str, nonce: int, reels: List[List[Tuple[str,int]]]) -> List[str]:
        h = ProvablyFairEngine.hmac_roll(server_seed, client_seed, nonce)
        out = []
        for ridx, reel in enumerate(reels):
            symbols = [s for s,_ in reel]
            weights = [w for _,w in reel]
            idx = ExtendedPF.weighted_index(h, ridx, weights)
            out.append(symbols[idx])
        return out

    @staticmethod
    def tower_path(server_seed: str, client_seed: str, nonce: int, levels: int, choices: int) -> List[int]:
        h = ProvablyFairEngine.hmac_roll(server_seed, client_seed, nonce)
        path = []
        for lvl in range(levels):
            u = ProvablyFairEngine.digest_to_uniform(h, offset=lvl % 4)
            path.append(int(u * choices))
        return path

    @staticmethod
    def keno_draw(server_seed: str, client_seed: str, nonce: int, pool=80, draw=20) -> List[int]:
        h = ProvablyFairEngine.hmac_roll(server_seed, client_seed, nonce)
        nums = list(range(1, pool+1))
        for i in range(pool-1,0,-1):
            u = ProvablyFairEngine.digest_to_uniform(h, offset=(pool - i) %4)
            j = int(u*(i+1))
            nums[i], nums[j] = nums[j], nums[i]
        return sorted(nums[:draw])

    @staticmethod
    def blackjack_shuffled_deck(server_seed: str, client_seed: str, nonce: int, decks: int=6) -> List[str]:
        ranks = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
        suits = ["S","H","D","C"]
        deck = [f"{r}{s}" for _ in range(decks) for r in ranks for s in suits]
        h = ProvablyFairEngine.hmac_roll(server_seed, client_seed, nonce)
        for i in range(len(deck)-1,0,-1):
            u = ProvablyFairEngine.digest_to_uniform(h, offset=i % 4)
            j = int(u*(i+1))
            deck[i], deck[j] = deck[j], deck[i]
        return deck

# ---------------------------------------------------------------------
# GAME INFRASTRUCTURE
# ---------------------------------------------------------------------

class GameError(Exception): pass

@dataclass
class BetResult:
    outcome: Dict[str, Any]
    payout: Decimal
    result_label: str

class AbstractGame:
    name: str = "abstract"
    edge: float = 0.01
    max_win: Decimal = Decimal(str(Config.MAX_WIN))
    def __init__(self, edge: Optional[float] = None):
        if edge is not None:
            self.edge = edge
    async def play(self, server_seed: str, client_seed: str, nonce: int, amount: Decimal, params: Dict[str, Any]) -> BetResult:
        raise NotImplementedError
    def apply_max_win(self, payout: Decimal) -> Decimal:
        return min(payout, self.max_win)

# (Basic games)
class DiceGame(AbstractGame):
    name="dice"
    async def play(self, server_seed, client_seed, nonce, amount, params):
        target = float(params.get("target", 50))
        if not (0.01 <= target < 100): raise GameError("Bad target")
        roll = ProvablyFairEngine.dice_roll(server_seed, client_seed, nonce)
        win = roll < target
        mult = (100/target)*(1 - self.edge)
        payout = amount * Decimal(str(mult)) if win else Decimal("0")
        payout = safe_floor_decimal(payout)
        return BetResult({"roll": round(roll,4), "target": target}, self.apply_max_win(payout), "win" if win else "lose")

class LimboGame(AbstractGame):
    name="limbo"
    async def play(self, server_seed, client_seed, nonce, amount, params):
        M = float(params.get("multiplier", 2))
        if M < 1.01 or M > 100000: raise GameError("Bad M")
        crash_point = ProvablyFairEngine.limbo_point(server_seed, client_seed, nonce)
        win = crash_point >= M
        payout = amount * Decimal(str(M)) * Decimal(str(1 - self.edge)) if win else Decimal("0")
        payout = safe_floor_decimal(payout)
        return BetResult({"crash_point": round(crash_point,4), "target_multiplier": M}, self.apply_max_win(payout), "win" if win else "lose")

class CoinFlipGame(AbstractGame):
    name="coinflip"
    async def play(self, server_seed, client_seed, nonce, amount, params):
        pick = params.get("pick","heads").lower()
        if pick not in ("heads","tails"):
            raise GameError("Pick heads/tails")
        result = ProvablyFairEngine.coin_flip(server_seed, client_seed, nonce)
        win = (pick==result)
        payout = amount * Decimal(str(2*(1 - self.edge))) if win else Decimal("0")
        payout = safe_floor_decimal(payout)
        return BetResult({"result": result, "pick": pick}, self.apply_max_win(payout), "win" if win else "lose")

class RouletteGame(AbstractGame):
    name="roulette"
    async def play(self, server_seed, client_seed, nonce, amount, params):
        bet_type = params.get("bet_type","straight")
        number = int(params.get("number",0))
        outcome_num = ProvablyFairEngine.roulette_number(server_seed, client_seed, nonce)
        win=False; payout=Decimal("0")
        if bet_type=="straight" and 0<=number<=36:
            win = (outcome_num==number)
            if win:
                payout = amount * Decimal("36") * Decimal(str(1 - self.edge))
        payout = safe_floor_decimal(payout)
        return BetResult({"number": outcome_num, "bet_type": bet_type, "pick": number}, self.apply_max_win(payout), "win" if win else "lose")

class CrashGame(AbstractGame):
    name="crash"
    async def play(self, *a, **k):
        raise GameError("Crash uses real-time round system.")

class MinesGame(AbstractGame):
    name="mines"
    async def play(self, server_seed, client_seed, nonce, amount, params):
        grid_size = int(params.get("grid_size",25))
        mines = int(params.get("mines",3))
        picks = int(params.get("picks",1))
        if not (1 <= mines < grid_size): raise GameError("Invalid mines")
        if not (1 <= picks < grid_size - mines): raise GameError("Invalid picks")
        mine_positions = ProvablyFairEngine.mines_positions(server_seed, client_seed, nonce, grid_size, mines)
        safe_positions = [p for p in range(grid_size) if p not in mine_positions]
        h = ProvablyFairEngine.hmac_roll(server_seed, client_seed, nonce)
        u = ProvablyFairEngine.digest_to_uniform(h)
        remaining_safes = len(safe_positions)
        remaining_total = grid_size
        survival_prob=1.0
        for _ in range(picks):
            survival_prob *= remaining_safes / remaining_total
            remaining_safes-=1; remaining_total-=1
        win = u < survival_prob
        mult = (1/survival_prob)*(1 - self.edge)
        payout = amount * Decimal(str(mult)) if win else Decimal("0")
        payout = safe_floor_decimal(payout)
        return BetResult(
            {"mine_positions": mine_positions,"picks": picks,"survival_prob": round(survival_prob,6)},
            self.apply_max_win(payout),
            "win" if win else "lose"
        )

class TowerGame(AbstractGame):
    name="tower"
    async def play(self, server_seed, client_seed, nonce, amount, params):
        levels = int(params.get("levels",5))
        choices = int(params.get("choices",3))
        cashout_level = int(params.get("cashout_level", levels))
        if not (1<=levels<=15 and 2<=choices<=5 and 1<=cashout_level<=levels):
            raise GameError("Bad params")
        path = ExtendedPF.tower_path(server_seed, client_seed, nonce, levels, choices)
        per_level_survival = (choices-1)/choices
        total_survival = per_level_survival ** cashout_level
        h = ProvablyFairEngine.hmac_roll(server_seed, client_seed, nonce)
        win = ProvablyFairEngine.digest_to_uniform(h) < total_survival
        mult = (1/total_survival)*(1 - self.edge)
        payout = amount * Decimal(str(mult)) if win else Decimal("0")
        payout = safe_floor_decimal(payout)
        return BetResult({"lose_path": path, "cashout_level": cashout_level, "survival_prob": round(total_survival,6)},
                         self.apply_max_win(payout),
                         "win" if win else "lose")

class KenoGame(AbstractGame):
    name="keno"
    async def play(self, server_seed, client_seed, nonce, amount, params):
        picks = params.get("picks",[1,2,3,4,5])
        if isinstance(picks,str):
            try: picks=json.loads(picks)
            except: raise GameError("Bad picks json")
        picks = list(sorted(set(int(p) for p in picks)))
        if not (1<=len(picks)<=15): raise GameError("Invalid pick count")
        if any(not (1<=p<=80) for p in picks): raise GameError("Pick range")
        draw = ExtendedPF.keno_draw(server_seed, client_seed, nonce)
        hits = len(set(picks)&set(draw))
        paytable = {
            1:{1:3},
            2:{2:10,1:1},
            3:{3:25,2:3,1:1},
            4:{4:100,3:10,2:2},
            5:{5:500,4:50,3:5,2:2},
            6:{6:1500,5:200,4:20,3:3},
            7:{7:5000,6:500,5:50,4:10,3:5},
            8:{8:10000,7:2000,6:200,5:30,4:5},
            9:{9:25000,8:4000,7:400,6:40,5:10},
            10:{10:50000,9:10000,8:1000,7:100,6:20,5:10}
        }
        table = paytable.get(len(picks),{})
        mult = Decimal("0")
        if hits in table:
            mult = Decimal(str(table[hits])) * Decimal(str(1 - self.edge))
        payout = safe_floor_decimal(amount * mult)
        return BetResult({"draw": draw, "picks": picks, "hits": hits}, self.apply_max_win(payout),
                         "win" if payout>0 else "lose")

class PokerHouseGame(AbstractGame):
    name="poker_house"
    async def play(self, server_seed, client_seed, nonce, amount, params):
        deck = ExtendedPF.blackjack_shuffled_deck(server_seed, client_seed, nonce, decks=1)
        player = deck[:5]; dealer = deck[5:10]
        def score(hand):
            # naive rank
            mapv = {"A":14,"K":13,"Q":12,"J":11,"10":10,"9":9,"8":8,"7":7,"6":6,"5":5,"4":4,"3":3,"2":2}
            ranks=[c[:-1] if c[:-1] else c[0] for c in hand]
            ints=[mapv[r] for r in ranks]; counts={}
            for v in ints: counts[v]=counts.get(v,0)+1
            groups=sorted(counts.values(),reverse=True)
            if groups==[4,1]: rtype=4
            elif groups==[3,2]: rtype=3
            elif groups==[3,1,1]: rtype=2
            elif groups in ([2,2,1],[2,1,1,1]): rtype=1
            else: rtype=0
            return (rtype, sorted(ints, reverse=True))
        ps=score(player); ds=score(dealer)
        if ps>ds: payout=amount*Decimal(str(2*(1 - self.edge))); result="win"
        elif ps==ds: payout=amount; result="push"
        else: payout=Decimal("0"); result="lose"
        payout=safe_floor_decimal(payout)
        return BetResult({"player":player,"dealer":dealer,"ps":ps,"ds":ds}, self.apply_max_win(payout), result)

# Slots w/ jackpot hook later
class SlotsGame(AbstractGame):
    name="slots"
    def __init__(self, edge=None):
        super().__init__(edge)
        self.reels = [
            [("🍒",30),("🍋",25),("⭐",15),("7️⃣",5),("🔔",10),("💎",5),("🍀",10)],
            [("🍒",30),("🍋",25),("⭐",15),("7️⃣",5),("🔔",10),("💎",5),("🍀",10)],
            [("🍒",30),("🍋",25),("⭐",15),("7️⃣",5),("🔔",10),("💎",5),("🍀",10)],
        ]
        self.paytable = {
            ("7️⃣","7️⃣","7️⃣"): 500,
            ("💎","💎","💎"): 200,
            ("🍀","🍀","🍀"): 100,
            ("⭐","⭐","⭐"): 50,
            ("🔔","🔔","🔔"): 25,
            ("🍒","🍒","🍒"): 10,
            ("🍋","🍋","🍋"): 8,
        }
    async def play(self, server_seed, client_seed, nonce, amount, params):
        landed = ExtendedPF.slots_reels(server_seed, client_seed, nonce, self.reels)
        base_mult = self.paytable.get(tuple(landed), 0)
        payout = amount * Decimal(str(base_mult)) * Decimal(str(1 - self.edge)) if base_mult>0 else Decimal("0")
        payout = safe_floor_decimal(payout)
        return BetResult({"reels": landed}, self.apply_max_win(payout), "win" if payout>0 else "lose")

# Blackjack (full interactive handled by sessions; single "blackjack_full" placeholder)
class FullBlackjackGame(AbstractGame):
    name="blackjack_full"
    async def play(self, *a, **k):
        raise GameError("Use interactive /bj_* commands.")

class PvpDiceEmojiGame(AbstractGame):
    name="pvp_dice_emoji"
    async def play(self,*a,**k): raise GameError("Use PvP flows")

class DartsPvpGame(AbstractGame):
    name="darts_pvp"
    async def play(self,*a,**k): raise GameError("Use PvP flows")

class HoldemHouseGame(AbstractGame):
    name="holdem_house"
    async def play(self, server_seed, client_seed, nonce, amount, params):
        # 5 community + 2 player vs dealer 2
        deck = ExtendedPF.blackjack_shuffled_deck(server_seed, client_seed, nonce, decks=1)
        player_hole = deck[:2]; dealer_hole=deck[2:4]; community=deck[4:9]
        def eval7(cards):
            # reuse from poker_house simple approach
            # ignore flush/straight complexities for brevity
            mapv={"A":14,"K":13,"Q":12,"J":11,"10":10,"9":9,"8":8,"7":7,"6":6,"5":5,"4":4,"3":3,"2":2}
            vals=sorted([mapv[c[:-1] if c[:-1] else c[0]] for c in cards], reverse=True)[:5]
            return sum(vals)
        pr = eval7(player_hole+community)
        dr = eval7(dealer_hole+community)
        if pr>dr: payout=amount*Decimal(str(2*(1 - self.edge))); result="win"
        elif pr==dr: payout=amount; result="push"
        else: payout=Decimal("0"); result="lose"
        payout = safe_floor_decimal(payout)
        return BetResult({"player_hole":player_hole,"dealer_hole":dealer_hole,"community":community}, self.apply_max_win(payout), result)

# Register games
GAME_REGISTRY: Dict[str, AbstractGame] = {
    "dice": DiceGame(edge=Config.HOUSE_EDGES["dice"]),
    "limbo": LimboGame(edge=Config.HOUSE_EDGES["limbo"]),
    "coinflip": CoinFlipGame(edge=Config.HOUSE_EDGES["coinflip"]),
    "roulette": RouletteGame(edge=Config.HOUSE_EDGES["roulette"]),
    "blackjack_full": FullBlackjackGame(edge=Config.HOUSE_EDGES["blackjack"]),
    "crash": CrashGame(edge=Config.HOUSE_EDGES["crash"]),
    "mines": MinesGame(edge=Config.HOUSE_EDGES["mines"]),
    "tower": TowerGame(edge=Config.HOUSE_EDGES["tower"]),
    "keno": KenoGame(edge=Config.HOUSE_EDGES["keno"]),
    "slots": SlotsGame(edge=Config.HOUSE_EDGES["slots"]),
    "poker_house": PokerHouseGame(edge=Config.HOUSE_EDGES["pvp_rake"]),
    "pvp_dice_emoji": PvpDiceEmojiGame(edge=Config.HOUSE_EDGES["pvp_rake"]),
    "darts_pvp": DartsPvpGame(edge=Config.HOUSE_EDGES["pvp_rake"]),
    "holdem_house": HoldemHouseGame(edge=0.03),
}

# ---------------------------------------------------------------------
# SERVICES
# ---------------------------------------------------------------------

class UserService:
    @staticmethod
    def get_or_create(session, tg_user: TgUser) -> User:
        u = session.execute(select(User).where(User.tg_id==tg_user.id)).scalar_one_or_none()
        if not u:
            u = User(
                tg_id=tg_user.id,
                username=tg_user.username
            )
            session.add(u); session.flush()
            seed_val = random_seed()
            sseed = ServerSeed(user_id=u.id, server_seed=seed_val, server_seed_hash=hash_seed(seed_val), active=True)
            session.add(sseed); session.flush()
            u.current_server_seed_id = sseed.id
            u.server_seed_hash = sseed.server_seed_hash
            session.commit()
        else:
            if tg_user.username and u.username != tg_user.username:
                u.username = tg_user.username
                session.commit()
        return u

    @staticmethod
    def rotate_server_seed(session, user: User) -> Tuple[str,str]:
        old_seed_obj = session.get(ServerSeed, user.current_server_seed_id)
        old_seed_obj.active=False
        old_seed_obj.rotated_at=now_utc()
        old_plain = old_seed_obj.server_seed
        new_seed = random_seed()
        new_seed_hash = hash_seed(new_seed)
        new_obj = ServerSeed(user_id=user.id, server_seed=new_seed, server_seed_hash=new_seed_hash, active=True)
        session.add(new_obj); session.flush()
        user.current_server_seed_id = new_obj.id
        user.server_seed_hash = new_seed_hash
        user.bets_count_in_seed = 0
        session.commit()
        return old_plain, new_seed_hash

class BetService:
    @staticmethod
    def place_bet(session, user: User, game_key: str, amount: Decimal, params: Dict[str,Any]) -> Bet:
        if user.frozen: raise GameError("Frozen")
        if user.self_excluded_until and user.self_excluded_until > now_utc():
            raise GameError("Self-excluded")
        amount = clamp_bet(amount)
        if amount > user.balance: raise GameError("Insufficient balance")
        if game_key not in GAME_REGISTRY: raise GameError("Game unsupported")
        seed_obj = session.get(ServerSeed, user.current_server_seed_id)
        server_seed = seed_obj.server_seed
        client_seed = user.client_seed
        nonce = user.bets_count_in_seed
        game = GAME_REGISTRY[game_key]

        user.balance -= amount
        async def async_play():
            return await game.play(server_seed, client_seed, nonce, amount, params)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            result: BetResult = asyncio.run(async_play())
        else:
            result: BetResult = loop.run_until_complete(async_play())

        payout = result.payout
        user.balance += payout
        user.bets_count_in_seed += 1
        user.lifetime_wager += amount
        user.xp += int(float(amount) * Config.XP_RATE)
        user.vip_tier = compute_vip_tier(float(user.lifetime_wager))

        bet = Bet(
            user_id=user.id,
            game=game_key,
            amount=amount,
            result=result.result_label,
            payout=payout,
            nonce=nonce,
            server_seed_id=seed_obj.id,
            client_seed_snapshot=client_seed,
            outcome_data=result.outcome
        )
        session.add(bet)
        if user.bets_count_in_seed >= Config.SEED_ROTATION_BETS:
            UserService.rotate_server_seed(session, user)
        session.commit()
        return bet

    @staticmethod
    def verify_bet(session, bet_id: int) -> Dict[str, Any]:
        bet = session.get(Bet, bet_id)
        if not bet: raise ValueError("Bet not found")
        seed_obj = session.get(ServerSeed, bet.server_seed_id)
        if seed_obj.active:
            return {
                "bet_id": bet.id, "status":"seed_not_revealed_yet",
                "server_seed_hash": seed_obj.server_seed_hash,
                "client_seed": bet.client_seed_snapshot,
                "nonce": bet.nonce,
                "game": bet.game
            }
        return {
            "bet_id": bet.id, "status":"verifiable",
            "server_seed": seed_obj.server_seed,
            "server_seed_hash": seed_obj.server_seed_hash,
            "client_seed": bet.client_seed_snapshot,
            "nonce": bet.nonce,
            "game": bet.game,
            "outcome_data": bet.outcome_data,
            "payout": str(bet.payout),
            "result": bet.result
        }

class WalletService:
    @staticmethod
    def create_deposit_address(session, user: User, chain: str) -> str:
        # Placeholder deterministic
        addr = "0x" + hashlib.sha256(f"{user.id}:{chain}:{time.time()}".encode()).hexdigest()[:40]
        return addr

    @staticmethod
    def record_deposit(session, user: User, chain: str, address: str, tx_hash: str, amount: Decimal):
        dep = DepositTx(
            user_id=user.id, chain=chain, address=address,
            tx_hash=tx_hash, amount=amount
        )
        session.add(dep); session.commit()
        return dep

    @staticmethod
    def credit_deposits(session):
        pending = session.execute(select(DepositTx).where(DepositTx.credited==False)).scalars().all()
        for dep in pending:
            dep.confirmations += 1
            if dep.confirmations >= Config.DEFAULT_CONFIRMATIONS:
                user = session.get(User, dep.user_id)
                user.balance += dep.amount
                dep.credited = True
        session.commit()

    @staticmethod
    def request_withdrawal(session, user: User, chain: str, address: str, amount: Decimal):
        if amount <= 0: raise ValueError("Bad amount")
        if amount > user.balance: raise ValueError("Insufficient balance")
        if not is_valid_evm_address(address): raise ValueError("Invalid address")
        user.balance -= amount
        wd = WithdrawalTx(user_id=user.id, chain=chain, address=address, amount=amount, status="pending")
        session.add(wd); session.commit()
        return wd

    @staticmethod
    def admin_process_withdrawal(session, wid: int, approve: bool, admin_id: int, tx_hash: Optional[str]=None, reason: Optional[str]=None):
        wd = session.get(WithdrawalTx, wid)
        if not wd or wd.status!="pending": raise ValueError("Not pending")
        if approve:
            wd.status="paid"
            wd.tx_hash = tx_hash or "MANUAL_TX"
            wd.admin_id = admin_id
        else:
            wd.status="denied"
            wd.reason=reason or "No reason"
            wd.admin_id=admin_id
            user = session.get(User, wd.user_id)
            user.balance += wd.amount
        session.commit()
        return wd

class RewardService:
    @staticmethod
    def accrue_periodic_rewards(session):
        now = now_utc()
        week_start = now - timedelta(days=7)
        month_start = now - timedelta(days=30)
        users = session.execute(select(User)).scalars().all()
        for u in users:
            # Weekly
            wk_ps = week_start.replace(hour=0,minute=0,second=0,microsecond=0)
            exists = session.execute(select(Reward).where(
                Reward.user_id==u.id, Reward.type=="weekly", Reward.period_start==wk_ps
            )).scalar_one_or_none()
            if not exists:
                wager7 = session.execute(
                    select(func.coalesce(func.sum(Bet.amount),0)).where(Bet.user_id==u.id, Bet.created_at>=week_start)
                ).scalar_one()
                if wager7>0:
                    amt = Decimal(str(wager7)) * Decimal(str(Config.WEEKLY_REWARD_RATE))
                    session.add(Reward(
                        user_id=u.id, type="weekly", period_start=wk_ps, period_end=now,
                        amount=safe_floor_decimal(amt), expires_at=now+timedelta(days=Config.REWARD_EXPIRE_DAYS)
                    ))
            # Monthly
            mo_ps = month_start.replace(hour=0,minute=0,second=0,microsecond=0)
            exists = session.execute(select(Reward).where(
                Reward.user_id==u.id, Reward.type=="monthly", Reward.period_start==mo_ps
            )).scalar_one_or_none()
            if not exists:
                wager30 = session.execute(
                    select(func.coalesce(func.sum(Bet.amount),0)).where(Bet.user_id==u.id, Bet.created_at>=month_start)
                ).scalar_one()
                if wager30>0:
                    amt = Decimal(str(wager30)) * Decimal(str(Config.MONTHLY_REWARD_RATE))
                    session.add(Reward(
                        user_id=u.id, type="monthly", period_start=mo_ps, period_end=now,
                        amount=safe_floor_decimal(amt), expires_at=now+timedelta(days=Config.REWARD_EXPIRE_DAYS)
                    ))
            # Rebate (weekly)
            net7 = session.execute(
                select(func.coalesce(func.sum(Bet.payout - Bet.amount),0)).where(Bet.user_id==u.id, Bet.created_at>=week_start)
            ).scalar_one()
            if net7 < 0:
                exists = session.execute(select(Reward).where(
                    Reward.user_id==u.id, Reward.type=="rebate", Reward.period_start==wk_ps
                )).scalar_one_or_none()
                if not exists:
                    amt = Decimal(str(-net7)) * Decimal(str(Config.REBATE_RATE_BASE))
                    session.add(Reward(
                        user_id=u.id, type="rebate", period_start=wk_ps, period_end=now,
                        amount=safe_floor_decimal(amt), expires_at=now+timedelta(days=Config.REWARD_EXPIRE_DAYS)
                    ))
        session.commit()

    @staticmethod
    def claim_reward(session, user: User, rid: int) -> Reward:
        r = session.get(Reward, rid)
        if not r or r.user_id!=user.id: raise ValueError("Reward not found")
        if r.claimed_at: raise ValueError("Already claimed")
        if r.expires_at < now_utc(): raise ValueError("Expired")
        user.balance += r.amount
        r.claimed_at = now_utc()
        session.commit()
        return r

# ---------------------------------------------------------------------
# BLACKJACK ENGINE (INTERACTIVE)
# ---------------------------------------------------------------------

def bj_card_value(card: str) -> int:
    rank = card[:-1] if card[:-1] else card[0]
    if rank in ("J","Q","K"): return 10
    if rank=="A": return 11
    return int(rank)

def bj_hand_totals(cards: List[str]) -> Tuple[int,bool]:
    total=0; aces=0
    for c in cards:
        v=bj_card_value(c)
        if v==11: aces+=1
        total+=v
    while total>21 and aces>0:
        total-=10; aces-=1
    soft = (aces>0)
    return total, soft

class BlackjackEngine:
    @staticmethod
    def start_round(server_seed, client_seed, nonce, decks=6):
        deck = ExtendedPF.blackjack_shuffled_deck(server_seed, client_seed, nonce, decks)
        player=[deck.pop(0), deck.pop(0)]
        dealer=[deck.pop(0), deck.pop(0)]
        return {
            "deck": deck,
            "player_hands": [player],
            "dealer_hand": dealer,
            "active_hand_index": 0,
            "completed": False,
            "doubled": [False],
            "bets":[1],
            "split_possible": player[0][:-1]==player[1][:-1],
            "stage": "player_action"
        }
    @staticmethod
    def hit(state, idx):
        deck=state["deck"]
        card=deck.pop(0)
        state["player_hands"][idx].append(card)
        total,_=bj_hand_totals(state["player_hands"][idx])
        if total>=21:
            BlackjackEngine.advance_hand(state)
    @staticmethod
    def stand(state):
        BlackjackEngine.advance_hand(state)
    @staticmethod
    def double(state, idx):
        if state["doubled"][idx]: return
        if len(state["player_hands"][idx])!=2: return
        state["doubled"][idx]=True
        state["bets"][idx]*=2
        BlackjackEngine.hit(state, idx)
    @staticmethod
    def split(state):
        if state["split_possible"] and len(state["player_hands"])==1:
            h=state["player_hands"][0]
            if h[0][:-1]==h[1][:-1]:
                deck=state["deck"]
                c1=h[0]; c2=h[1]
                state["player_hands"]=[[c1,deck.pop(0)],[c2,deck.pop(0)]]
                state["doubled"]=[False,False]
                state["bets"]=[1,1]
                state["active_hand_index"]=0
                state["split_possible"]=False
    @staticmethod
    def advance_hand(state):
        state["active_hand_index"]+=1
        if state["active_hand_index"]>=len(state["player_hands"]):
            BlackjackEngine.dealer_play(state)
            state["completed"]=True
            state["stage"]="settle"
        else:
            state["stage"]="player_action"
    @staticmethod
    def dealer_play(state):
        dealer=state["dealer_hand"]
        deck=state["deck"]
        total,soft = bj_hand_totals(dealer)
        while total<17 or (total==17 and soft):
            dealer.append(deck.pop(0))
            total,soft = bj_hand_totals(dealer)
    @staticmethod
    def settle(state, edge: float):
        dealer_total,_ = bj_hand_totals(state["dealer_hand"])
        total_payout=Decimal("0")
        results=[]
        details=[]
        for idx,hand in enumerate(state["player_hands"]):
            bet_unit=Decimal(str(state["bets"][idx]))
            p_total,_=bj_hand_totals(hand)
            if len(hand)==2 and p_total==21 and not state["split_possible"]:
                d_bj = (len(state["dealer_hand"])==2 and dealer_total==21)
                if d_bj:
                    results.append("push"); payout_mult=bet_unit
                else:
                    results.append("blackjack")
                    payout_mult=bet_unit*Decimal("2.5")*Decimal(str(1 - edge))
            else:
                if p_total>21:
                    results.append("bust"); payout_mult=Decimal("0")
                elif dealer_total>21:
                    results.append("win"); payout_mult=bet_unit*Decimal("2")*Decimal(str(1 - edge))
                else:
                    if p_total>dealer_total:
                        results.append("win"); payout_mult=bet_unit*Decimal("2")*Decimal(str(1 - edge))
                    elif p_total==dealer_total:
                        results.append("push"); payout_mult=bet_unit
                    else:
                        results.append("lose"); payout_mult=Decimal("0")
            total_payout += payout_mult
            details.append({"hand": idx,"player_cards":hand,"player_total":p_total,"dealer_total":dealer_total,"outcome":results[-1],"payout_mult":float(payout_mult)})
        return {
            "result": ",".join(results),
            "payout_multiplier": total_payout,
            "details": details,
            "dealer_hand": state["dealer_hand"],
            "player_hands": state["player_hands"]
        }

# ---------------------------------------------------------------------
# INTERACTIVE MINES SESSION MANAGER
# ---------------------------------------------------------------------

class MinesSessionManager:
    GRID_SIZE=25
    MAX_MINES=15
    @staticmethod
    def start(session_db, user: User, bet: Decimal, mines: int) -> GameSession:
        if mines<1 or mines>MinesSessionManager.MAX_MINES: raise GameError("Invalid mines")
        if bet>user.balance: raise GameError("Insufficient balance")
        user.balance -= bet
        seed_obj = session_db.get(ServerSeed, user.current_server_seed_id)
        mp = ProvablyFairEngine.mines_positions(seed_obj.server_seed, user.client_seed, user.bets_count_in_seed, MinesSessionManager.GRID_SIZE, mines)
        state={
            "bet": str(bet),"mines": mines,"revealed":[],"mine_positions": mp,"active":True,
            "picks":0,"server_seed_id": seed_obj.id,"nonce": user.bets_count_in_seed
        }
        gs=GameSession(user_id=user.id, game="mines_interactive", state_json=state)
        session_db.add(gs); session_db.commit()
        return gs
    @staticmethod
    def pick(session_db, gs: GameSession, user: User, cell: int):
        st=gs.state_json
        if not st["active"]: raise GameError("Ended")
        if cell in st["revealed"]: raise GameError("Already revealed")
        if not (0<=cell<MinesSessionManager.GRID_SIZE): raise GameError("Bad cell")
        st["revealed"].append(cell); st["picks"]+=1
        if cell in st["mine_positions"]:
            st["active"]=False
            gs.state_json=st; gs.updated_at=now_utc()
            session_db.commit()
            return {"status":"bust","revealed": st["revealed"]}
        # Compute suggested cashout
        remaining_safe = MinesSessionManager.GRID_SIZE - len(st["mine_positions"])
        surv=1.0; rs=remaining_safe; rt=MinesSessionManager.GRID_SIZE
        for k in range(st["picks"]):
            surv *= rs/rt; rs-=1; rt-=1
        edge = Config.HOUSE_EDGES["mines"]
        mult = (1/max(1e-12,surv))*(1 - edge)
        gs.state_json=st; gs.updated_at=now_utc()
        session_db.commit()
        return {"status":"continue","revealed": st["revealed"],"multiplier": round(mult,6),"suggested_cashout": str(safe_floor_decimal(Decimal(st["bet"])*Decimal(str(mult))))}
    @staticmethod
    def cashout(session_db, gs: GameSession, user: User):
        st=gs.state_json
        if not st["active"]: raise GameError("Already ended")
        bet=Decimal(st["bet"]); picks=st["picks"]
        if picks==0:
            payout=bet
        else:
            remaining_safe = MinesSessionManager.GRID_SIZE - len(st["mine_positions"])
            surv=1.0; rs=remaining_safe; rt=MinesSessionManager.GRID_SIZE
            for k in range(picks):
                surv *= rs/rt; rs-=1; rt-=1
            edge = Config.HOUSE_EDGES["mines"]
            mult=(1/max(1e-12,surv))*(1 - edge)
            payout=bet*Decimal(str(mult))
        payout=safe_floor_decimal(payout)
        user.balance += payout
        st["active"]=False
        gs.active=False
        gs.state_json=st; gs.updated_at=now_utc()
        # record bet
        b=Bet(
            user_id=user.id, game="mines_interactive", amount=bet, result="cashout", payout=payout,
            nonce=st["nonce"], server_seed_id=st["server_seed_id"], client_seed_snapshot=user.client_seed,
            outcome_data={"revealed": st["revealed"],"mine_positions": st["mine_positions"],"picks": picks}
        )
        session_db.add(b)
        user.bets_count_in_seed += 1
        if user.bets_count_in_seed >= Config.SEED_ROTATION_BETS:
            UserService.rotate_server_seed(session_db, user)
        session_db.commit()
        return {"status":"cashed_out","payout": str(payout)}

# ---------------------------------------------------------------------
# PVP ENGINE (Dice/Darts)
# ---------------------------------------------------------------------

class PvPEngine:
    @staticmethod
    def create_match(session_db, user: User, game: str, entry_fee: Decimal, best_of: int, rake_rate: float):
        if entry_fee<=0: raise GameError("Entry fee >0")
        if best_of not in (1,3,5,7): raise GameError("best_of invalid")
        if user.balance < entry_fee: raise GameError("Insufficient")
        user.balance -= entry_fee
        m=PvPMatch(
            game=game, creator_user_id=user.id, entry_fee=entry_fee, pot=entry_fee,
            rake=Decimal("0"), best_of=best_of, state_json={"scores":{"creator":0,"opponent":0},"round":0}, status="waiting"
        )
        session_db.add(m); session_db.commit()
        return m
    @staticmethod
    def join_match(session_db, mid: int, user: User):
        m=session_db.get(PvPMatch, mid)
        if not m or m.status!="waiting": raise GameError("Not joinable")
        if m.creator_user_id==user.id: raise GameError("Own match")
        if user.balance < m.entry_fee: raise GameError("Insufficient")
        user.balance -= m.entry_fee
        m.opponent_user_id = user.id
        m.pot += m.entry_fee
        m.status="active"
        m.updated_at=now_utc()
        session_db.commit()
        return m
    @staticmethod
    def record_round(session_db, mid: int):
        m=session_db.get(PvPMatch, mid)
        if not m or m.status!="active": raise GameError("Match not active")
        sj=m.state_json
        sj["round"]+=1
        c_roll=random.randint(1,6)
        o_roll=random.randint(1,6)
        if c_roll>o_roll: sj["scores"]["creator"]+=1
        elif o_roll>c_roll: sj["scores"]["opponent"]+=1
        best_of=m.best_of
        need=(best_of//2)+1
        if sj["scores"]["creator"]>=need or sj["scores"]["opponent"]>=need:
            # finish
            if sj["scores"]["creator"]>sj["scores"]["opponent"]:
                winner_id=m.creator_user_id
            else:
                winner_id=m.opponent_user_id
            rake_rate=Config.HOUSE_EDGES["pvp_rake"]
            rake_amt=safe_floor_decimal(m.pot*Decimal(str(rake_rate)))
            payout=m.pot - rake_amt
            w=session_db.get(User, winner_id)
            w.balance += payout
            m.rake=rake_amt
            m.winner_user_id=winner_id
            m.status="completed"
        m.state_json=sj; m.updated_at=now_utc()
        session_db.commit()
        return m, c_roll, o_roll
    @staticmethod
    def forfeit(session_db, mid: int, user: User):
        m=session_db.get(PvPMatch, mid)
        if not m or m.status not in ("waiting","active"):
            raise GameError("Cannot forfeit")
        if user.id not in (m.creator_user_id, m.opponent_user_id):
            raise GameError("Not participant")
        if m.status=="waiting":
            if user.id==m.creator_user_id:
                # refund
                cu=session_db.get(User, m.creator_user_id)
                cu.balance += m.entry_fee
                m.status="cancelled"
        else:
            other = m.opponent_user_id if user.id==m.creator_user_id else m.creator_user_id
            rake_rate=Config.HOUSE_EDGES["pvp_rake"]
            rake_amt=safe_floor_decimal(m.pot*Decimal(str(rake_rate)))
            payout=m.pot - rake_amt
            ou=session_db.get(User, other)
            ou.balance += payout
            m.rake=rake_amt
            m.status="completed"
            m.winner_user_id=other
        session_db.commit()
        return m

# ---------------------------------------------------------------------
# CRASH SYSTEM (REAL-TIME)
# ---------------------------------------------------------------------

class CrashState:
    ACTIVE=False
    ROUND_ID=0
    START_TIME=0.0
    CRASH_POINT=0.0
    SEED_TUPLE=None
    BETS: Dict[int, Dict[str, Any]] = {}
    NEXT_BET_ID=1
    LOCK_BETS=False

CRASH_BET_LOCK = asyncio.Lock()
CRASH_CASHOUT_LOCK = asyncio.Lock()

def current_crash_multiplier() -> float:
    if not CrashState.ACTIVE: return 1.0
    elapsed = time.time()-CrashState.START_TIME
    mult = 1.0 + elapsed * Config.CRASH_GROWTH_RATE * 25
    return min(mult, CrashState.CRASH_POINT)

async def crash_round_loop():
    while True:
        try:
            CrashState.ACTIVE=True
            CrashState.ROUND_ID+=1
            server_seed=random_seed()
            client_seed=random_seed(16)
            nonce=0
            crash_pt = ProvablyFairEngine.crash_multiplier(server_seed, client_seed, nonce)
            crash_pt = max(1.01, min(crash_pt, Config.CRASH_MAX_MULT))
            CrashState.CRASH_POINT=crash_pt
            CrashState.START_TIME=time.time()
            CrashState.SEED_TUPLE=(server_seed, client_seed, nonce)
            CrashState.BETS={}
            CrashState.NEXT_BET_ID=1
            CrashState.LOCK_BETS=False
            logger.info(f"[CRASH] Round {CrashState.ROUND_ID} start crash={crash_pt:.2f}x")
            await asyncio.sleep(5)
            CrashState.LOCK_BETS=True
            while True:
                mult=current_crash_multiplier()
                # Auto-cashouts
                with SessionLocal() as session:
                    for bid,b in CrashState.BETS.items():
                        if b["cashed"]: continue
                        if mult >= b["auto_cash"]:
                            user=session.get(User,b["user_id"])
                            if user:
                                payout=Decimal(str(b["amount"])) * Decimal(str(b["auto_cash"] * (1 - Config.HOUSE_EDGES["crash"])))
                                payout=safe_floor_decimal(payout)
                                user.balance += payout
                                b["cashed"]=True
                                b["payout"]=float(payout)
                                seed_obj=session.get(ServerSeed, user.current_server_seed_id)
                                bet_rec=Bet(
                                    user_id=user.id, game="crash", amount=Decimal(str(b["amount"])),
                                    result="win", payout=payout, nonce=user.bets_count_in_seed,
                                    server_seed_id=seed_obj.id, client_seed_snapshot=user.client_seed,
                                    outcome_data={"round_id":CrashState.ROUND_ID,"crash_point":crash_pt,"auto_cash":b["auto_cash"],"final_mult":mult}
                                )
                                session.add(bet_rec)
                                user.bets_count_in_seed+=1
                                if user.bets_count_in_seed>=Config.SEED_ROTATION_BETS:
                                    UserService.rotate_server_seed(session, user)
                    session.commit()
                if mult >= CrashState.CRASH_POINT:
                    # Crash -> settle losers
                    with SessionLocal() as session:
                        for bid,b in CrashState.BETS.items():
                            if not b["cashed"]:
                                user=session.get(User,b["user_id"])
                                if not user: continue
                                seed_obj=session.get(ServerSeed,user.current_server_seed_id)
                                bet_rec=Bet(
                                    user_id=user.id, game="crash", amount=Decimal(str(b["amount"])),
                                    result="lose", payout=Decimal("0"), nonce=user.bets_count_in_seed,
                                    server_seed_id=seed_obj.id, client_seed_snapshot=user.client_seed,
                                    outcome_data={"round_id":CrashState.ROUND_ID,"crash_point":CrashState.CRASH_POINT,"auto_cash":b["auto_cash"],"final_mult":CrashState.CRASH_POINT}
                                )
                                session.add(bet_rec)
                                user.bets_count_in_seed+=1
                                if user.bets_count_in_seed>=Config.SEED_ROTATION_BETS:
                                    UserService.rotate_server_seed(session, user)
                        session.commit()
                    logger.info(f"[CRASH] Crashed at {CrashState.CRASH_POINT:.2f}x")
                    break
                await asyncio.sleep(1)
            CrashState.ACTIVE=False
            await asyncio.sleep(5)
        except Exception:
            logger.exception("Crash loop error")
            await asyncio.sleep(5)

# ---------------------------------------------------------------------
# JACKPOT (SLOTS)
# ---------------------------------------------------------------------

def contribute_jackpot(session, amount: Decimal):
    jp=session.query(SlotJackpot).first()
    if not jp:
        jp=SlotJackpot(pool=Decimal("0"))
        session.add(jp); session.flush()
    contrib=safe_floor_decimal(amount*Decimal(str(Config.JACKPOT_CONTRIB_RATE)))
    jp.pool+=contrib
    session.commit()
    return contrib

def try_hit_jackpot(session, server_seed, client_seed, nonce, user: User):
    h=ProvablyFairEngine.hmac_roll(server_seed, client_seed, nonce)
    fragment=h[:8]
    val=int(fragment,16)/0xFFFFFFFF
    if val < Config.JACKPOT_HIT_CHANCE:
        jp=session.query(SlotJackpot).first()
        if jp and jp.pool>0:
            win=jp.pool
            user.balance += win
            jp.pool=Decimal("0")
            session.commit()
            return win
    return Decimal("0")

# Monkey wrap slots
_original_slots_play = GAME_REGISTRY["slots"].play
async def slots_play_jackpot(server_seed, client_seed, nonce, amount, params):
    with SessionLocal() as session:
        contribute_jackpot(session, amount)
        res=await _original_slots_play(server_seed, client_seed, nonce, amount, params)
        user_id=params.get("user_id")
        if user_id:
            user=session.get(User,user_id)
            if user:
                hit_amt=try_hit_jackpot(session, server_seed, client_seed, nonce, user)
                if hit_amt>0:
                    res.outcome["jackpot_hit"]=True
                    res.outcome["jackpot_amount"]=str(hit_amt)
        return res
GAME_REGISTRY["slots"].play = slots_play_jackpot

# ---------------------------------------------------------------------
# HOLDEM MULTI-SEAT (Simplified)
# ---------------------------------------------------------------------

class HoldemEngine:
    @staticmethod
    def create_table(session, bb: Decimal, max_players: int, rake_rate: float):
        if not (2<=max_players<=9): raise GameError("2-9 players")
        tbl=HoldemTable(big_blind=bb,max_players=max_players,rake_rate=Decimal(str(rake_rate)),status="waiting",state_json={})
        session.add(tbl); session.commit()
        return tbl

    @staticmethod
    def join_table(session, table_id: int, user: User, buyin: Decimal):
        tbl=session.get(HoldemTable, table_id)
        if not tbl or tbl.status not in ("waiting","active"): raise GameError("Not joinable")
        seats=session.execute(select(HoldemSeat).where(HoldemSeat.table_id==table_id, HoldemSeat.active==True)).scalars().all()
        if len(seats)>=tbl.max_players: raise GameError("Full")
        if buyin<=0 or buyin>user.balance: raise GameError("Bad buyin")
        user.balance -= buyin
        idxs={s.seat_index for s in seats}
        seat_index=0
        while seat_index in idxs: seat_index+=1
        seat=HoldemSeat(table_id=table_id,user_id=user.id,seat_index=seat_index,stack=buyin)
        session.add(seat); session.commit()
        return seat

    @staticmethod
    def leave_table(session, table_id: int, user: User):
        seat=session.execute(select(HoldemSeat).where(HoldemSeat.table_id==table_id, HoldemSeat.user_id==user.id, HoldemSeat.active==True)).scalar_one_or_none()
        if not seat: raise GameError("Not seated")
        seat.active=False
        user.balance+=seat.stack
        seat.stack=Decimal("0")
        session.commit()

    @staticmethod
    def start_hand(session, table_id: int):
        tbl=session.get(HoldemTable, table_id)
        if not tbl or tbl.status=="closed": raise GameError("Bad table")
        seats=session.execute(select(HoldemSeat).where(HoldemSeat.table_id==table_id, HoldemSeat.active==True)).scalars().all()
        if len(seats)<2: raise GameError("Need 2+")
        deck=ExtendedPF.blackjack_shuffled_deck(random_seed(),random_seed(16),random.randint(0,10),decks=1)
        random.shuffle(seats)
        ordering=[s.seat_index for s in seats]
        hole={}
        for s in seats: hole[str(s.seat_index)]=[deck.pop(0), deck.pop(0)]
        tbl.status="active"
        state={
            "hand_number": tbl.state_json.get("hand_number",0)+1,
            "phase":"preflop",
            "deck": deck,
            "ordering": ordering,
            "hole": hole,
            "community": [],
            "pot": "0",
            "bets": {},
            "acted": [],
            "last_action": None
        }
        tbl.state_json=state
        tbl.updated_at=now_utc()
        session.commit()
        return state

    @staticmethod
    def action(session, table_id: int, user: User, action: str, amount: Optional[Decimal]):
        tbl=session.get(HoldemTable, table_id)
        if not tbl or tbl.status!="active": raise GameError("Not active")
        state=tbl.state_json
        seat=session.execute(select(HoldemSeat).where(HoldemSeat.table_id==table_id, HoldemSeat.user_id==user.id, HoldemSeat.active==True)).scalar_one_or_none()
        if not seat: raise GameError("Not seated")
        if seat.seat_index in state["acted"]: raise GameError("Already acted")
        pot=Decimal(state["pot"]); current_bets={int(k):Decimal(v) for k,v in state["bets"].items()}
        call_amount=max(current_bets.values()) if current_bets else Decimal("0")
        player_bet=current_bets.get(seat.seat_index, Decimal("0"))
        to_call=call_amount - player_bet

        if action=="fold":
            seat.active=False
            state["acted"].append(seat.seat_index)
        elif action=="check":
            if to_call>0: raise GameError("Cannot check")
            state["acted"].append(seat.seat_index)
        elif action=="call":
            if to_call<=0: raise GameError("Nothing to call")
            pay=min(seat.stack,to_call)
            seat.stack-=pay; pot+=pay
            state["bets"][str(seat.seat_index)]=str(player_bet+pay)
            state["acted"].append(seat.seat_index)
        elif action=="bet":
            if call_amount>0: raise GameError("Bet exists")
            if not amount or amount<=0: raise GameError("Need amount")
            betval=min(seat.stack,amount)
            seat.stack-=betval; pot+=betval
            state["bets"][str(seat.seat_index)]=str(betval)
            state["acted"].append(seat.seat_index)
        elif action=="raise":
            if call_amount==0: raise GameError("No bet to raise")
            if not amount or amount<=call_amount: raise GameError("Raise must exceed")
            raise_needed=amount - player_bet
            raise_paid=min(seat.stack, raise_needed)
            seat.stack-=raise_paid; pot+=raise_paid
            state["bets"][str(seat.seat_index)]=str(player_bet+raise_paid)
            state["acted"].append(seat.seat_index)
        elif action=="allin":
            all_in=seat.stack
            if all_in<=0: raise GameError("Empty stack")
            seat.stack=Decimal("0")
            pot+=all_in
            state["bets"][str(seat.seat_index)]=str(player_bet+all_in)
            state["acted"].append(seat.seat_index)
        else:
            raise GameError("Unknown action")

        state["pot"]=str(pot)
        state["last_action"]={"seat": seat.seat_index,"action":action}
        tbl.state_json=state; tbl.updated_at=now_utc()
        session.commit()
        HoldemEngine._maybe_advance(session,tbl)
        return state

    @staticmethod
    def _active_seats(session, table_id:int):
        return session.execute(select(HoldemSeat).where(HoldemSeat.table_id==table_id, HoldemSeat.active==True)).scalars().all()

    @staticmethod
    def _maybe_advance(session, tbl: HoldemTable):
        state=tbl.state_json
        phase=state["phase"]
        seats=HoldemEngine._active_seats(session, tbl.id)
        if len(seats)<=1:
            HoldemEngine._showdown(session, tbl)
            return
        acted=set(state["acted"]); active_idxs={s.seat_index for s in seats}
        if active_idxs.issubset(acted):
            if phase=="preflop":
                deck=state["deck"]
                state["community"]=[deck.pop(0), deck.pop(0), deck.pop(0)]
                state["phase"]="flop"
            elif phase=="flop":
                state["community"].append(state["deck"].pop(0))
                state["phase"]="turn"
            elif phase=="turn":
                state["community"].append(state["deck"].pop(0))
                state["phase"]="river"
            elif phase=="river":
                HoldemEngine._showdown(session, tbl)
                return
            state["acted"]=[]; state["bets"]={}; state["last_action"]=None
            tbl.state_json=state; tbl.updated_at=now_utc()
            session.commit()

    @staticmethod
    def _hand_rank_7(cards: List[str]) -> Tuple[int,List[int]]:
        # extremely simplified (re-using partial logic from earlier)
        ranks_map={"A":14,"K":13,"Q":12,"J":11,"10":10,"9":9,"8":8,"7":7,"6":6,"5":5,"4":4,"3":3,"2":2}
        vals=sorted([ranks_map[c[:-1] if c[:-1] else c[0]] for c in cards], reverse=True)[:5]
        return (sum(vals), vals)

    @staticmethod
    def _showdown(session, tbl: HoldemTable):
        state=tbl.state_json
        community=state.get("community",[])
        seats=HoldemEngine._active_seats(session, tbl.id)
        if not seats:
            tbl.status="closed"
            session.commit(); return
        if len(seats)==1:
            winner=seats[0]
            pot=Decimal(state["pot"])
            rake_rate=float(tbl.rake_rate)
            rake=safe_floor_decimal(pot*Decimal(str(rake_rate)))
            payout=pot - rake
            winner.stack += payout
            state["showdown"]={"winner": winner.seat_index, "payout": str(payout),"rake": str(rake)}
            tbl.status="waiting"
        else:
            best=(-1,[])
            winners=[]
            for s in seats:
                hole = state["hole"][str(s.seat_index)]
                r=HoldemEngine._hand_rank_7(hole+community)
                if r>best:
                    best=r; winners=[s]
                elif r==best:
                    winners.append(s)
            pot=Decimal(state["pot"])
            rake_rate=float(tbl.rake_rate)
            rake=safe_floor_decimal(pot*Decimal(str(rake_rate)))
            remain=pot - rake
            share=safe_floor_decimal(remain/Decimal(len(winners)))
            for w in winners: w.stack += share
            state["showdown"]={"winners":[w.seat_index for w in winners],"payout_each":str(share),"rake": str(rake)}
            tbl.status="waiting"
        state["phase"]="showdown"
        tbl.state_json=state; tbl.updated_at=now_utc()
        session.commit()

# ---------------------------------------------------------------------
# ANALYTICS / AUDIT / COLLUSION
# ---------------------------------------------------------------------

def audit_log(session, category: str, actor_user_id: Optional[int], target_user_id: Optional[int], reference: Optional[str], data: Dict[str,Any]):
    log=AuditLog(category=category,actor_user_id=actor_user_id,target_user_id=target_user_id,reference=reference,data=data)
    session.add(log); session.commit()

def compute_rtp_report(session, days: int=30) -> Dict[str, Any]:
    cutoff=now_utc()-timedelta(days=days)
    rows=session.execute(
        select(Bet.game, func.count(Bet.id), func.coalesce(func.sum(Bet.amount),0), func.coalesce(func.sum(Bet.payout),0))
        .where(Bet.created_at>=cutoff).group_by(Bet.game)
    ).all()
    report={}
    for game,cnt,wager,paid in rows:
        wager=Decimal(wager); paid=Decimal(paid)
        rtp=float(paid/wager) if wager>0 else 0
        report[game]={"bets": int(cnt),"wagered": str(wager),"paid": str(paid),"rtp": round(rtp,5)}
    return report

def detect_collusion_patterns(session):
    # Shared device fingerprints
    fp_rows=session.execute(
        select(UserDeviceFingerprint.fingerprint_hash, func.count(UserDeviceFingerprint.user_id))
        .group_by(UserDeviceFingerprint.fingerprint_hash)
        .having(func.count(UserDeviceFingerprint.user_id)>1)
    ).all()
    # Mutual tipping heuristic (simplified)
    tip_rows=session.execute(
        select(Tip.from_user, Tip.to_user, func.count(Tip.id))
        .group_by(Tip.from_user, Tip.to_user)
        .having(func.count(Tip.id)>=5)
    ).all()
    return {
        "shared_fingerprints": [{"fingerprint":f,"count":c} for f,c in fp_rows],
        "mutual_tipping_candidates":[list(r) for r in tip_rows]
    }

# ---------------------------------------------------------------------
# CALLBACK SIGNING (UPGRADED)
# ---------------------------------------------------------------------

CALLBACK_EXP_SECONDS=180
CALLBACK_NONCE_CACHE: Dict[str,float] = {}
CALLBACK_NONCE_TTL=300

def cleanup_callback_nonce_cache():
    now=time.time()
    expired=[k for k,v in CALLBACK_NONCE_CACHE.items() if now - v > CALLBACK_NONCE_TTL]
    for k in expired: CALLBACK_NONCE_CACHE.pop(k,None)

def is_replay(nonce: str) -> bool:
    cleanup_callback_nonce_cache()
    if nonce in CALLBACK_NONCE_CACHE: return True
    CALLBACK_NONCE_CACHE[nonce]=time.time()
    return False

def build_signed_callback(action: str, **payload) -> str:
    body={"a":action,"ts": int(time.time()),"nonce": random_seed(16),"d": payload}
    canonical=json.dumps(body, sort_keys=True, separators=(",",":"))
    sig=hmac.new(Config.CALLBK_SECRET.encode(), canonical.encode(), hashlib.sha256).hexdigest()
    body["sig"]=sig
    return json.dumps(body, separators=(",",":"))

def verify_signed_callback(raw: str) -> Optional[Dict[str,Any]]:
    try:
        body=json.loads(raw)
        sig=body.get("sig","")
        unsigned={k:v for k,v in body.items() if k!="sig"}
        canonical=json.dumps(unsigned, sort_keys=True, separators=(",",":"))
        expected=hmac.new(Config.CALLBK_SECRET.encode(), canonical.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected): return None
        if time.time() - unsigned["ts"] > CALLBACK_EXP_SECONDS: return None
        if is_replay(unsigned["nonce"]): return None
        return unsigned
    except: return None

# ---------------------------------------------------------------------
# REST API + WEBSOCKETS
# ---------------------------------------------------------------------

api_app: Optional[FastAPI] = FastAPI(title="CasinoBot API", version="0.1") if FastAPI else None

if api_app:
    @api_app.get("/api/health")
    def api_health():
        return {"status":"ok","time": datetime.utcnow().isoformat()}

    @api_app.get("/api/profile")
    def api_profile(tg_id: int, token: str):
        payload=verify_session_token(token)
        if not payload or payload.get("tg")!=tg_id:
            raise HTTPException(401, "Invalid token")
        with SessionLocal() as session:
            u=session.execute(select(User).where(User.tg_id==tg_id)).scalar_one_or_none()
            if not u: raise HTTPException(404,"User not found")
            return {"tg_id": tg_id,"balance": str(u.balance),"vip_tier": u.vip_tier,"xp": u.xp,"lifetime_wager": str(u.lifetime_wager)}

    @api_app.get("/api/bet/{bet_id}")
    def api_bet(bet_id: int, token: str):
        payload=verify_session_token(token)
        if not payload: raise HTTPException(401,"Invalid token")
        with SessionLocal() as session:
            bet=session.get(Bet, bet_id)
            if not bet: raise HTTPException(404,"Bet not found")
            user=session.execute(select(User).where(User.tg_id==payload["tg"])).scalar_one_or_none()
            if not user or user.id!=bet.user_id: raise HTTPException(403,"Forbidden")
            seed_obj=session.get(ServerSeed, bet.server_seed_id)
            if seed_obj.active:
                return {"bet_id": bet.id,"status":"seed_not_revealed","server_seed_hash": seed_obj.server_seed_hash,
                        "client_seed": bet.client_seed_snapshot,"nonce": bet.nonce}
            return {"bet_id": bet.id,"status":"verifiable","server_seed": seed_obj.server_seed,
                    "server_seed_hash": seed_obj.server_seed_hash,"client_seed": bet.client_seed_snapshot,"nonce": bet.nonce,
                    "outcome_data": bet.outcome_data,"payout": str(bet.payout),"result": bet.result}

    @api_app.get("/api/jackpot")
    def api_jackpot():
        with SessionLocal() as session:
            jp=session.query(SlotJackpot).first()
            return {"pool": str(jp.pool if jp else 0)}

    # WebSockets
    if FastAPI:
        @api_app.websocket("/ws/crash")
        async def ws_crash(ws: WebSocket):
            await ws.accept()
            try:
                while True:
                    mult=current_crash_multiplier() if CrashState.ACTIVE else 1.0
                    await ws.send_json({"round": CrashState.ROUND_ID,"active": CrashState.ACTIVE,"multiplier": mult,"target": CrashState.CRASH_POINT})
                    await asyncio.sleep(1)
            except WebSocketDisconnect:
                pass
            except Exception:
                logger.exception("WS crash error")

        @api_app.websocket("/ws/jackpot")
        async def ws_jackpot(ws: WebSocket):
            await ws.accept()
            try:
                while True:
                    with SessionLocal() as session:
                        jp=session.query(SlotJackpot).first()
                        await ws.send_json({"jackpot": str(jp.pool if jp else 0),"time": datetime.utcnow().isoformat()})
                    await asyncio.sleep(5)
            except WebSocketDisconnect:
                pass
            except Exception:
                logger.exception("WS jackpot error")

async def start_rest_api():
    if not api_app:
        logger.warning("FastAPI not installed; skipping REST server.")
        return
    import uvicorn
    config=uvicorn.Config(api_app, host="0.0.0.0", port=int(os.getenv("API_PORT","8000")), log_level="info", loop="asyncio")
    server=uvicorn.Server(config)
    await server.serve()

# ---------------------------------------------------------------------
# COMMAND HANDLERS
# ---------------------------------------------------------------------

HELP_TEXT = (
"/start - Main menu\n"
"/help - This help\n"
"/balance - Show balance\n"
"/games - List games\n"
"/bet <game> <amount> <params_json>\n"
"/deposit <chain>\n"
"/withdraw <chain> <address> <amount>\n"
"/profile\n"
"/seed [rotate]\n"
"/verify <bet_id>\n"
"/tip <@user|id> <amount>\n"
"/rain <amount> <count>\n"
"/rewards - Claimable rewards\n"
"/selfexclude <days>\n"
"/cooldown <minutes>\n"
"/admin ... (admins)\n"
"/bj_start [amount]\n"
"/mines_start <bet> <mines>\n"
"/crash_bet <amount> <auto_cash>\n"
"/crash_cashout <bet_id>\n"
"/session_token\n"
"/rtp_report [days]\n"
+ responsible_gambling_notice()
)

MAIN_MENU_BUTTONS=[
    [InlineKeyboardButton("🎲 Games", callback_data=build_signed_callback("menu_games"))],
    [InlineKeyboardButton("💰 Deposit", callback_data=build_signed_callback("menu_deposit")),
     InlineKeyboardButton("🏦 Withdraw", callback_data=build_signed_callback("menu_withdraw"))],
    [InlineKeyboardButton("👤 Profile", callback_data=build_signed_callback("menu_profile")),
     InlineKeyboardButton("🏅 VIP & Rewards", callback_data=build_signed_callback("menu_vip"))],
    [InlineKeyboardButton("ℹ️ Provably Fair", callback_data=build_signed_callback("menu_pf")),
     InlineKeyboardButton("❓ Help", callback_data=build_signed_callback("menu_help"))],
]

async def send_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, user: User):
    await update.effective_message.reply_text(
        f"Welcome {update.effective_user.first_name or 'Player'}!\n"
        f"Balance: {format_amount(user.balance)}\nVIP: {user.vip_tier}\n"
        f"Server Seed Hash: {user.server_seed_hash[:16]}...\n"
        f"{responsible_gambling_notice()}",
        reply_markup=InlineKeyboardMarkup(MAIN_MENU_BUTTONS)
    )

# Basic user commands
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with SessionLocal() as session:
        user=UserService.get_or_create(session, update.effective_user)
    await send_main_menu(update, context, user)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)

async def balance_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with SessionLocal() as session:
        user=UserService.get_or_create(session, update.effective_user)
        await update.message.reply_text(f"Balance: {format_amount(user.balance)} VIP: {user.vip_tier} XP: {user.xp}")

async def games_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Games: " + ", ".join(sorted(GAME_REGISTRY.keys())))

async def bet_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message: return
    parts=update.message.text.strip().split(" ",3)
    if len(parts)<3:
        await update.message.reply_text("Usage: /bet <game> <amount> <params_json>")
        return
    game=parts[1].lower()
    try: amount=Decimal(parts[2])
    except: await update.message.reply_text("Bad amount"); return
    params={}
    if len(parts)>=4:
        try: params=json.loads(parts[3])
        except: await update.message.reply_text("Bad JSON for params"); return
    with SessionLocal() as session:
        user=UserService.get_or_create(session, update.effective_user)
        if game=="slots":
            params.setdefault("user_id", user.id)
        try:
            bet=BetService.place_bet(session,user,game,amount,params)
            await update.message.reply_text(
                f"Bet #{bet.id} {game} result={bet.result} payout={format_amount(bet.payout)} outcome={bet.outcome_data} /verify {bet.id}"
            )
        except GameError as ge:
            await update.message.reply_text(f"Bet error: {ge}")
        except Exception:
            logger.exception("Bet exception")
            await update.message.reply_text("Internal error")

async def seed_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with SessionLocal() as session:
        user=UserService.get_or_create(session, update.effective_user)
        if context.args and context.args[0].lower()=="rotate":
            old_seed,new_hash=UserService.rotate_server_seed(session,user)
            await update.message.reply_text(f"Rotated. Old seed: {old_seed}\nNew hash: {new_hash}")
        else:
            await update.message.reply_text(f"Current Server Seed Hash: {user.server_seed_hash}\nRotate: /seed rotate")

async def verify_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /verify <bet_id>")
        return
    try: bid=int(context.args[0])
    except: await update.message.reply_text("Bad bet id"); return
    with SessionLocal() as session:
        try:
            info=BetService.verify_bet(session,bid)
            await update.message.reply_text(json.dumps(info, indent=2)[:3800])
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

async def deposit_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chain="ETH"
    if context.args: chain=context.args[0].upper()
    if chain not in Config.SUPPORTED_CHAINS:
        await update.message.reply_text("Unsupported chain")
        return
    with SessionLocal() as session:
        user=UserService.get_or_create(session, update.effective_user)
        addr=WalletService.create_deposit_address(session, user, chain)
        await update.message.reply_text(
            f"Deposit Address {chain}: {addr}\nRequires {Config.DEFAULT_CONFIRMATIONS} confirmations."
        )

async def withdraw_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args)<3:
        await update.message.reply_text("Usage: /withdraw <chain> <address> <amount>")
        return
    chain=context.args[0].upper()
    address=context.args[1]
    try: amount=Decimal(context.args[2])
    except: await update.message.reply_text("Bad amount"); return
    if chain not in Config.SUPPORTED_CHAINS:
        await update.message.reply_text("Unsupported chain"); return
    with SessionLocal() as session:
        user=UserService.get_or_create(session, update.effective_user)
        try:
            wd=WalletService.request_withdrawal(session,user,chain,address,amount)
            await update.message.reply_text(f"Withdrawal request #{wd.id} pending admin approval.")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

async def profile_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with SessionLocal() as session:
        user=UserService.get_or_create(session, update.effective_user)
        bet_count=session.execute(select(func.count(Bet.id)).where(Bet.user_id==user.id)).scalar_one()
        net=session.execute(select(func.coalesce(func.sum(Bet.payout - Bet.amount),0)).where(Bet.user_id==user.id)).scalar_one()
        await update.message.reply_text(
            f"Profile:\nBalance: {format_amount(user.balance)}\nVIP: {user.vip_tier}\nXP: {user.xp}\n"
            f"Lifetime Wager: {format_amount(user.lifetime_wager)}\nBets: {bet_count}\nNet Profit: {format_amount(net)}\nClient Seed: {user.client_seed[:12]}..."
        )

async def tip_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args)<2:
        await update.message.reply_text("Usage: /tip <@user|id> <amount>")
        return
    target=context.args[0]
    try: amt=Decimal(context.args[1])
    except: await update.message.reply_text("Bad amount"); return
    if amt<=0:
        await update.message.reply_text("Amount must be >0"); return
    with SessionLocal() as session:
        sender=UserService.get_or_create(session, update.effective_user)
        if sender.balance<amt:
            await update.message.reply_text("Insufficient balance"); return
        if target.startswith("@"):
            uname=target[1:].lower()
            recipient=session.execute(select(User).where(func.lower(User.username)==uname)).scalar_one_or_none()
        else:
            try: tid=int(target)
            except: recipient=None
            else:
                recipient=session.execute(select(User).where(User.tg_id==tid)).scalar_one_or_none()
        if not recipient:
            await update.message.reply_text("User not found (they must /start)")
            return
        sender.balance -= amt
        recipient.balance += amt
        session.add(Tip(from_user=sender.id,to_user=recipient.id,amount=amt))
        session.commit()
        await update.message.reply_text(f"Tipped {format_amount(amt)} to {recipient.username or recipient.tg_id}")

async def rain_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args)<2:
        await update.message.reply_text("Usage: /rain <amount> <count>")
        return
    try:
        total=Decimal(context.args[0])
        count=int(context.args[1])
    except:
        await update.message.reply_text("Bad args"); return
    if total<=0 or count<=0:
        await update.message.reply_text("Invalid values"); return
    with SessionLocal() as session:
        user=UserService.get_or_create(session, update.effective_user)
        if user.balance<total:
            await update.message.reply_text("Insufficient balance"); return
        recents=session.execute(select(User.id).order_by(User.id.desc()).limit(100)).scalars().all()
        recents=[r for r in recents if r!=user.id]
        if not recents:
            await update.message.reply_text("No recipients")
            return
        import random
        chosen=random.sample(recents, min(count,len(recents)))
        per=safe_floor_decimal(total/Decimal(len(chosen)))
        user.balance -= per*Decimal(len(chosen))
        for cid in chosen:
            u=session.get(User,cid)
            u.balance += per
        session.add(Rain(from_user=user.id,total_amount=per*Decimal(len(chosen)),recipients_count=len(chosen),completed=True))
        session.commit()
        await update.message.reply_text(f"Rained {format_amount(per*len(chosen))} among {len(chosen)} users.")

async def rewards_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with SessionLocal() as session:
        user=UserService.get_or_create(session, update.effective_user)
        unclaimed=session.execute(select(Reward).where(
            Reward.user_id==user.id, Reward.claimed_at.is_(None), Reward.expires_at>now_utc()
        )).scalars().all()
        if not unclaimed:
            await update.message.reply_text("No unclaimed rewards.")
            return
        kb=[]
        lines=[]
        for r in unclaimed:
            lines.append(f"{r.type.title()} #{r.id} Amt: {format_amount(r.amount)} Exp: {r.expires_at.date()}")
            kb.append([InlineKeyboardButton(f"Claim {r.type.title()} #{r.id}", callback_data=build_signed_callback("claim_reward", rid=r.id))])
        await update.message.reply_text("Rewards:\n"+"\n".join(lines), reply_markup=InlineKeyboardMarkup(kb))

async def selfexclude_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /selfexclude <days>")
        return
    try: days=int(context.args[0])
    except: await update.message.reply_text("Bad number"); return
    if not (Config.SELF_EXCLUSION_DAYS_MIN <= days <= Config.SELF_EXCLUSION_DAYS_MAX):
        await update.message.reply_text("Out of range")
        return
    with SessionLocal() as session:
        user=UserService.get_or_create(session, update.effective_user)
        user.self_excluded_until=now_utc()+timedelta(days=days)
        session.commit()
        await update.message.reply_text(f"Self-exclusion until {user.self_excluded_until}")

async def cooldown_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /cooldown <minutes>")
        return
    try: minutes=int(context.args[0])
    except: await update.message.reply_text("Bad minutes"); return
    if not (1<=minutes<=1440): await update.message.reply_text("Range 1-1440"); return
    with SessionLocal() as session:
        user=UserService.get_or_create(session, update.effective_user)
        user.self_excluded_until=now_utc()+timedelta(minutes=minutes)
        session.commit()
        await update.message.reply_text(f"Cooldown until {user.self_excluded_until}")

# Blackjack interactive commands
async def bj_start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with SessionLocal() as session:
        user=UserService.get_or_create(session, update.effective_user)
        bet_amt=Decimal("0.001")
        if context.args:
            try: bet_amt=Decimal(context.args[0])
            except: pass
        if bet_amt>user.balance:
            await update.message.reply_text("Insufficient balance"); return
        seed_obj=session.get(ServerSeed, user.current_server_seed_id)
        state=BlackjackEngine.start_round(seed_obj.server_seed, user.client_seed, user.bets_count_in_seed)
        user.balance -= bet_amt
        state["base_bet"]=str(bet_amt)
        gs=GameSession(user_id=user.id, game="blackjack_full", state_json=state)
        session.add(gs); session.commit()
        await update.message.reply_text(
            f"Blackjack session {gs.id}. Your {state['player_hands'][0]} Dealer [{state['dealer_hand'][0]}, ?]\n"
            f"/bj_hit {gs.id} /bj_stand {gs.id} /bj_double {gs.id} /bj_split {gs.id}"
        )

async def bj_hit_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /bj_hit <session_id>"); return
    sid=int(context.args[0])
    with SessionLocal() as session:
        gs=session.get(GameSession,sid)
        if not gs or gs.game!="blackjack_full" or not gs.active:
            await update.message.reply_text("Not active")
            return
        user=UserService.get_or_create(session, update.effective_user)
        if gs.user_id!=user.id: await update.message.reply_text("Not yours"); return
        st=gs.state_json
        if st["stage"]!="player_action":
            await update.message.reply_text("Not your turn"); return
        BlackjackEngine.hit(st, st["active_hand_index"])
        gs.state_json=st; gs.updated_at=now_utc(); session.commit()
        if st.get("completed"):
            outcome=BlackjackEngine.settle(st, Config.HOUSE_EDGES["blackjack"])
            base_bet=Decimal(st["base_bet"])
            payout=safe_floor_decimal(base_bet * outcome["payout_multiplier"])
            user.balance += payout
            bet_rec=Bet(user_id=user.id, game="blackjack_full", amount=base_bet, result=outcome["result"],
                        payout=payout, nonce=user.bets_count_in_seed,
                        server_seed_id=user.current_server_seed_id, client_seed_snapshot=user.client_seed,
                        outcome_data=outcome["details"])
            session.add(bet_rec)
            user.bets_count_in_seed+=1
            if user.bets_count_in_seed>=Config.SEED_ROTATION_BETS:
                UserService.rotate_server_seed(session,user)
            gs.active=False; session.commit()
            await update.message.reply_text(
                f"Dealer {st['dealer_hand']} Player {st['player_hands']}\nOutcome {outcome['result']} Payout {format_amount(payout)}"
            )
        else:
            await update.message.reply_text(
                f"Hand {st['active_hand_index']+1}: {st['player_hands'][st['active_hand_index']]} Dealer [{st['dealer_hand'][0]}, ?]"
            )

async def bj_stand_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args: await update.message.reply_text("Usage: /bj_stand <session_id>"); return
    sid=int(context.args[0])
    with SessionLocal() as session:
        gs=session.get(GameSession,sid)
        if not gs or gs.game!="blackjack_full" or not gs.active:
            await update.message.reply_text("Not active"); return
        user=UserService.get_or_create(session, update.effective_user)
        if gs.user_id!=user.id: await update.message.reply_text("Not yours"); return
        st=gs.state_json
        if st["stage"]!="player_action":
            await update.message.reply_text("Not your turn"); return
        BlackjackEngine.stand(st)
        gs.state_json=st; gs.updated_at=now_utc(); session.commit()
        if st.get("completed"):
            outcome=BlackjackEngine.settle(st, Config.HOUSE_EDGES["blackjack"])
            base_bet=Decimal(st["base_bet"])
            payout=safe_floor_decimal(base_bet * outcome["payout_multiplier"])
            user.balance += payout
            bet_rec=Bet(user_id=user.id, game="blackjack_full", amount=base_bet, result=outcome["result"],
                        payout=payout, nonce=user.bets_count_in_seed,
                        server_seed_id=user.current_server_seed_id, client_seed_snapshot=user.client_seed,
                        outcome_data=outcome["details"])
            session.add(bet_rec)
            user.bets_count_in_seed+=1
            if user.bets_count_in_seed>=Config.SEED_ROTATION_BETS:
                UserService.rotate_server_seed(session,user)
            gs.active=False; session.commit()
            await update.message.reply_text(
                f"Dealer {st['dealer_hand']} Player {st['player_hands']} => {outcome['result']} Payout {format_amount(payout)}"
            )
        else:
            await update.message.reply_text("Next hand or action.")

async def bj_double_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args: await update.message.reply_text("Usage: /bj_double <session_id>"); return
    sid=int(context.args[0])
    with SessionLocal() as session:
        gs=session.get(GameSession,sid)
        if not gs or gs.game!="blackjack_full" or not gs.active: await update.message.reply_text("Inactive"); return
        user=UserService.get_or_create(session, update.effective_user)
        st=gs.state_json
        idx=st["active_hand_index"]
        base_bet=Decimal(st["base_bet"])
        if user.balance < base_bet:
            await update.message.reply_text("Insufficient balance for double"); return
        user.balance -= base_bet
        BlackjackEngine.double(st, idx)
        gs.state_json=st; gs.updated_at=now_utc(); session.commit()
        if st.get("completed"):
            outcome=BlackjackEngine.settle(st, Config.HOUSE_EDGES["blackjack"])
            payout=safe_floor_decimal(base_bet * outcome["payout_multiplier"])
            user.balance += payout
            bet_rec=Bet(user_id=user.id, game="blackjack_full", amount=base_bet, result=outcome["result"],
                        payout=payout, nonce=user.bets_count_in_seed,
                        server_seed_id=user.current_server_seed_id, client_seed_snapshot=user.client_seed,
                        outcome_data=outcome["details"])
            session.add(bet_rec)
            user.bets_count_in_seed+=1
            if user.bets_count_in_seed>=Config.SEED_ROTATION_BETS:
                UserService.rotate_server_seed(session,user)
            gs.active=False; session.commit()
            await update.message.reply_text(f"Double complete: {outcome['result']} Payout {format_amount(payout)}")
        else:
            await update.message.reply_text("Action updated after double.")

async def bj_split_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args: await update.message.reply_text("Usage: /bj_split <session_id>"); return
    sid=int(context.args[0])
    with SessionLocal() as session:
        gs=session.get(GameSession,sid)
        if not gs or gs.game!="blackjack_full" or not gs.active: await update.message.reply_text("Inactive"); return
        user=UserService.get_or_create(session, update.effective_user)
        st=gs.state_json
        if len(st["player_hands"])>1:
            await update.message.reply_text("Split not available")
            return
        base_bet=Decimal(st["base_bet"])
        if user.balance < base_bet:
            await update.message.reply_text("Insufficient to split"); return
        user.balance -= base_bet
        BlackjackEngine.split(st)
        gs.state_json=st; gs.updated_at=now_utc(); session.commit()
        await update.message.reply_text(f"Split done. Hands: {st['player_hands']}")

# Interactive Mines
async def mines_start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args)<2:
        await update.message.reply_text("Usage: /mines_start <bet> <mines>")
        return
    try:
        bet=Decimal(context.args[0]); mines=int(context.args[1])
    except:
        await update.message.reply_text("Invalid args"); return
    with SessionLocal() as session:
        user=UserService.get_or_create(session, update.effective_user)
        try:
            gs=MinesSessionManager.start(session,user,bet,mines)
            await update.message.reply_text(
                f"Mines session {gs.id} started bet {format_amount(bet)} mines={mines}. "
                f"/mines_pick {gs.id} <cell> /mines_cashout {gs.id}"
            )
        except GameError as ge:
            await update.message.reply_text(f"Error: {ge}")

async def mines_pick_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args)<2:
        await update.message.reply_text("Usage: /mines_pick <session_id> <cell>")
        return
    sid=int(context.args[0]); cell=int(context.args[1])
    with SessionLocal() as session:
        gs=session.get(GameSession,sid)
        if not gs or gs.game!="mines_interactive" or not gs.active:
            await update.message.reply_text("Not active"); return
        user=UserService.get_or_create(session, update.effective_user)
        if gs.user_id!=user.id: await update.message.reply_text("Not yours"); return
        try:
            res=MinesSessionManager.pick(session,gs,user,cell)
            await update.message.reply_text(json.dumps(res)[:3800])
        except GameError as ge:
            await update.message.reply_text(f"Error: {ge}")

async def mines_cashout_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /mines_cashout <session_id>")
        return
    sid=int(context.args[0])
    with SessionLocal() as session:
        gs=session.get(GameSession,sid)
        if not gs or gs.game!="mines_interactive" or not gs.active:
            await update.message.reply_text("Not active"); return
        user=UserService.get_or_create(session, update.effective_user)
        if gs.user_id!=user.id: await update.message.reply_text("Not yours"); return
        try:
            res=MinesSessionManager.cashout(session,gs,user)
            await update.message.reply_text(json.dumps(res))
        except GameError as ge:
            await update.message.reply_text(f"Error: {ge}")

# Crash commands
async def crash_bet_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args)<2:
        await update.message.reply_text("Usage: /crash_bet <amount> <auto_cash>")
        return
    try:
        amount=Decimal(context.args[0])
        auto=float(context.args[1])
    except:
        await update.message.reply_text("Bad args")
        return
    if not CrashState.ACTIVE or CrashState.LOCK_BETS:
        await update.message.reply_text("Betting closed")
        return
    if auto<1.01 or auto>Config.CRASH_MAX_MULT:
        await update.message.reply_text("Bad auto cashout"); return
    async with CRASH_BET_LOCK:
        with SessionLocal() as session:
            user=UserService.get_or_create(session, update.effective_user)
            if amount>user.balance:
                await update.message.reply_text("Insufficient")
                return
            if amount<Decimal(str(Config.MIN_BET)) or amount>Decimal(str(Config.MAX_BET)):
                await update.message.reply_text("Bet outside limits")
                return
            user.balance -= amount
            bid=CrashState.NEXT_BET_ID; CrashState.NEXT_BET_ID+=1
            CrashState.BETS[bid]={"user_id": user.id,"amount": float(amount),"auto_cash": auto,"cashed": False,"payout":0.0}
            session.commit()
        await update.message.reply_text(f"Crash bet #{bid} accepted round {CrashState.ROUND_ID} auto {auto}x")

async def crash_cashout_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /crash_cashout <bet_id>")
        return
    try: bid=int(context.args[0])
    except: await update.message.reply_text("Bad id"); return
    if not CrashState.ACTIVE:
        await update.message.reply_text("No active round")
        return
    async with CRASH_CASHOUT_LOCK:
        mult=current_crash_multiplier()
        if mult>=CrashState.CRASH_POINT:
            await update.message.reply_text("Already crashed")
            return
        with SessionLocal() as session:
            b=CrashState.BETS.get(bid)
            if not b:
                await update.message.reply_text("Bet not found"); return
            user=UserService.get_or_create(session, update.effective_user)
            if b["user_id"]!=user.id:
                await update.message.reply_text("Not your bet"); return
            if b["cashed"]:
                await update.message.reply_text("Already cashed")
                return
            payout=Decimal(str(b["amount"])) * Decimal(str(mult*(1 - Config.HOUSE_EDGES["crash"])))
            payout=safe_floor_decimal(payout)
            user.balance += payout
            b["cashed"]=True
            b["payout"]=float(payout)
            seed_obj=session.get(ServerSeed, user.current_server_seed_id)
            bet_rec=Bet(
                user_id=user.id, game="crash", amount=Decimal(str(b["amount"])),
                result="win", payout=payout, nonce=user.bets_count_in_seed,
                server_seed_id=seed_obj.id, client_seed_snapshot=user.client_seed,
                outcome_data={"round_id": CrashState.ROUND_ID,"crash_point": CrashState.CRASH_POINT,"manual_cash_mult": mult}
            )
            session.add(bet_rec)
            user.bets_count_in_seed+=1
            if user.bets_count_in_seed>=Config.SEED_ROTATION_BETS:
                UserService.rotate_server_seed(session,user)
            session.commit()
        await update.message.reply_text(f"Cashed out bet {bid} at {mult:.2f}x payout {format_amount(payout)}")

# PvP commands
async def pvp_create_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args)<3:
        await update.message.reply_text("Usage: /pvp_create <game:pvp_dice_emoji|darts_pvp> <entry_fee> <best_of>")
        return
    game=context.args[0]
    try: fee=Decimal(context.args[1]); best=int(context.args[2])
    except: await update.message.reply_text("Bad args"); return
    if game not in ("pvp_dice_emoji","darts_pvp"):
        await update.message.reply_text("Unsupported")
        return
    with SessionLocal() as session:
        user=UserService.get_or_create(session, update.effective_user)
        try:
            m=PvPEngine.create_match(session,user,game,fee,best,Config.HOUSE_EDGES["pvp_rake"])
            await update.message.reply_text(f"Match #{m.id} created /pvp_join {m.id}")
        except GameError as ge:
            await update.message.reply_text(f"Error: {ge}")

async def pvp_join_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args: await update.message.reply_text("Usage: /pvp_join <match_id>"); return
    mid=int(context.args[0])
    with SessionLocal() as session:
        user=UserService.get_or_create(session, update.effective_user)
        try:
            m=PvPEngine.join_match(session,mid,user)
            await update.message.reply_text(f"Joined #{mid}. /pvp_round {mid}")
        except GameError as ge:
            await update.message.reply_text(f"Error: {ge}")

async def pvp_round_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args: await update.message.reply_text("Usage: /pvp_round <match_id>"); return
    mid=int(context.args[0])
    with SessionLocal() as session:
        m=session.get(PvPMatch,mid)
        if not m or m.status!="active": await update.message.reply_text("Not active"); return
        user=UserService.get_or_create(session, update.effective_user)
        if user.id not in (m.creator_user_id,m.opponent_user_id):
            await update.message.reply_text("Not participant"); return
        try:
            m,c_roll,o_roll=PvPEngine.record_round(session,mid)
            scores=m.state_json["scores"]
            if m.status=="completed":
                await update.message.reply_text(
                    f"Round: creator {c_roll} opponent {o_roll}. Completed. Winner={m.winner_user_id} Pot={format_amount(m.pot)} Rake={format_amount(m.rake)}"
                )
            else:
                await update.message.reply_text(f"Rolled: creator {c_roll} opponent {o_roll} Scores: {scores}")
        except GameError as ge:
            await update.message.reply_text(f"Error: {ge}")

async def pvp_forfeit_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args: await update.message.reply_text("Usage: /pvp_forfeit <match_id>"); return
    mid=int(context.args[0])
    with SessionLocal() as session:
        user=UserService.get_or_create(session, update.effective_user)
        try:
            m=PvPEngine.forfeit(session,mid,user)
            await update.message.reply_text(f"Forfeited match {mid} status={m.status}")
        except GameError as ge:
            await update.message.reply_text(f"Error: {ge}")

# Hold'em commands
async def holdem_create_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args)<2:
        await update.message.reply_text("Usage: /holdem_create <big_blind> <max_players>")
        return
    try: bb=Decimal(context.args[0]); mp=int(context.args[1])
    except: await update.message.reply_text("Bad args"); return
    with SessionLocal() as session:
        tbl=HoldemEngine.create_table(session,bb,mp,Config.HOUSE_EDGES["pvp_rake"])
        await update.message.reply_text(f"Table {tbl.id} created. /holdem_join {tbl.id} <buyin>")

async def holdem_join_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args)<2:
        await update.message.reply_text("Usage: /holdem_join <table_id> <buyin>")
        return
    table_id=int(context.args[0])
    try: buy=Decimal(context.args[1])
    except: await update.message.reply_text("Bad buyin"); return
    with SessionLocal() as session:
        user=UserService.get_or_create(session, update.effective_user)
        try:
            seat=HoldemEngine.join_table(session,table_id,user,buy)
            await update.message.reply_text(f"Joined table {table_id} seat={seat.seat_index} stack={format_amount(seat.stack)}")
        except GameError as ge:
            await update.message.reply_text(f"Error: {ge}")

async def holdem_leave_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args: await update.message.reply_text("Usage: /holdem_leave <table_id>"); return
    table_id=int(context.args[0])
    with SessionLocal() as session:
        user=UserService.get_or_create(session, update.effective_user)
        try:
            HoldemEngine.leave_table(session,table_id,user)
            await update.message.reply_text("Left table.")
        except GameError as ge:
            await update.message.reply_text(f"Error: {ge}")

async def holdem_start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args: await update.message.reply_text("Usage: /holdem_start <table_id>"); return
    table_id=int(context.args[0])
    with SessionLocal() as session:
        try:
            st=HoldemEngine.start_hand(session, table_id)
            await update.message.reply_text(f"Hand {st['hand_number']} started phase {st['phase']}")
        except GameError as ge:
            await update.message.reply_text(f"Error: {ge}")

async def holdem_action_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args)<2:
        await update.message.reply_text("Usage: /holdem_action <table_id> <action> [amount]")
        return
    table_id=int(context.args[0])
    action=context.args[1].lower()
    amount=None
    if len(context.args)>=3:
        try: amount=Decimal(context.args[2])
        except: await update.message.reply_text("Bad amount"); return
    with SessionLocal() as session:
        user=UserService.get_or_create(session, update.effective_user)
        try:
            st=HoldemEngine.action(session,table_id,user,action,amount)
            await update.message.reply_text(f"Phase={st['phase']} Pot={st['pot']} Last={st['last_action']}")
        except GameError as ge:
            await update.message.reply_text(f"Error: {ge}")

async def table_list_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with SessionLocal() as session:
        rows=session.execute(select(HoldemTable).order_by(HoldemTable.id.desc()).limit(20)).scalars().all()
        if not rows:
            await update.message.reply_text("No tables")
            return
        lines=[f"#{t.id} status={t.status} players={len(t.state_json.get('hole',{}))} bb={t.big_blind}" for t in rows]
        await update.message.reply_text("\n".join(lines))

# RTP / Session token / Fingerprint / Audit / Exposure / Collusion
async def rtp_report_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with SessionLocal() as session:
        days=30
        if context.args:
            try: days=int(context.args[0])
            except: pass
        rpt=compute_rtp_report(session, days)
        await update.message.reply_text("RTP Report:\n"+json.dumps(rpt, indent=2)[:3800])

async def session_token_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    token=generate_session_token(update.effective_user.id, ttl_minutes=120)
    await update.message.reply_text(f"Session token (120m): {token}")

async def fingerprint_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /fingerprint <hash>")
        return
    fp=context.args[0]
    if len(fp)<16:
        await update.message.reply_text("Too short")
        return
    with SessionLocal() as session:
        user=UserService.get_or_create(session, update.effective_user)
        existing=session.execute(select(UserDeviceFingerprint).where(
            UserDeviceFingerprint.user_id==user.id, UserDeviceFingerprint.fingerprint_hash==fp
        )).scalar_one_or_none()
        if existing:
            existing.last_seen=now_utc(); existing.usage_count+=1
        else:
            session.add(UserDeviceFingerprint(user_id=user.id,fingerprint_hash=fp))
        session.commit()
        await update.message.reply_text("Fingerprint recorded.")

async def audit_tail_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    n=10
    if context.args:
        try: n=int(context.args[0])
        except: pass
    n=max(1,min(100,n))
    with SessionLocal() as session:
        rows=session.execute(select(AuditLog).order_by(AuditLog.id.desc()).limit(n)).scalars().all()
        lines=[f"{r.id} {r.category} a={r.actor_user_id} t={r.target_user_id} ref={r.reference} at={r.created_at.isoformat()}" for r in rows]
        await update.message.reply_text("Audit tail:\n"+"\n".join(lines))

async def exposure_snapshot_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("Not authorized"); return
    with SessionLocal() as session:
        total_bal=session.execute(select(func.coalesce(func.sum(User.balance),0))).scalar_one()
        jp=session.query(SlotJackpot).first()
        snap=ExposureSnapshot(total_user_balances=total_bal, jackpot_pool=jp.pool if jp else Decimal("0"))
        session.add(snap); session.commit()
        await update.message.reply_text(
            f"Exposure Snapshot id={snap.id} user_bal={format_amount(total_bal)} jackpot={format_amount(snap.jackpot_pool)}"
        )

async def collusion_report_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("Not authorized"); return
    with SessionLocal() as session:
        data=detect_collusion_patterns(session)
        await update.message.reply_text("Collusion Report:\n"+json.dumps(data, indent=2)[:3800])

async def kyc_flag_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with SessionLocal() as session:
        user=UserService.get_or_create(session, update.effective_user)
        user.kyc_flag=True
        session.commit()
        await update.message.reply_text("KYC flag set.")

# Simulations / tests
async def simulate_dice_ev_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    iters=5000
    if context.args:
        try: iters=int(context.args[0])
        except: pass
    game=GAME_REGISTRY["dice"]
    server_seed=random_seed(); client_seed=random_seed()
    bet=Decimal("1"); total=Decimal("0")
    wins=0
    for i in range(iters):
        roll=ProvablyFairEngine.dice_roll(server_seed, client_seed, i)
        win = roll<50
        if win:
            wins+=1
            mult=(100/50)*(1 - game.edge)
            total += bet*Decimal(str(mult)) - bet
        else:
            total -= bet
    ev=total/Decimal(iters)
    await update.message.reply_text(f"Dice EV/bet ~ {ev} wins {wins}/{iters}")

async def simulate_limbo_ev_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    iters=5000
    if context.args:
        try: iters=int(context.args[0])
        except: pass
    game=GAME_REGISTRY["limbo"]
    server_seed=random_seed(); client_seed=random_seed()
    bet=Decimal("1"); target=2.0
    total=Decimal("0")
    for i in range(iters):
        crash=ProvablyFairEngine.limbo_point(server_seed, client_seed, i)
        if crash>=target:
            total += bet*Decimal(str(target*(1 - game.edge))) - bet
        else:
            total -= bet
    ev=total/Decimal(iters)
    await update.message.reply_text(f"Limbo EV/bet ~ {ev}")

async def selftest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    trials=2000
    server_seed=random_seed(); client_seed=random_seed()
    wins=0
    for i in range(trials):
        if ProvablyFairEngine.dice_roll(server_seed, client_seed, i) < 50:
            wins+=1
    await update.message.reply_text(f"Empirical dice <50: {wins/trials:.4f}")

async def selftest_property_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    trials=500
    server_seed=random_seed(); client_seed=random_seed()
    seen=set(); dup=0
    for i in range(trials):
        roll=ProvablyFairEngine.dice_roll(server_seed, client_seed, i)
        if roll in seen: dup+=1
        seen.add(roll)
    await update.message.reply_text(f"PF uniqueness test duplicates={dup} of {trials}")

# Admin core command
ADMIN_HELP = (
    "Admin:\n"
    "list_users <limit>\nfreeze <tg_id>\nunfreeze <tg_id>\nsetedge <game> <edge>\nsetlimit <min|max|maxwin> <val>\n"
    "rotate_seed_user <tg_id>\napprove_withdraw <id> <txhash>\ndeny_withdraw <id> <reason...>\n"
    "jackpot_add <amt>\nkyc_flag <tg_id>\nkyc_clear <tg_id>\nregions_block <CC,CC>\nregions_unblock <CC,...>\n"
    "pvp_list\ncancel_match <id>\nsession_list\nend_session <id>\nexposure_snapshot\ncollusion_report\n"
)

async def admin_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("Not authorized")
        return
    if not context.args:
        await update.message.reply_text(ADMIN_HELP)
        return
    sub=context.args[0].lower()
    with SessionLocal() as session:
        if sub=="list_users":
            limit=10
            if len(context.args)>1 and context.args[1].isdigit():
                limit=int(context.args[1])
            users=session.execute(select(User).order_by(User.id.desc()).limit(limit)).scalars().all()
            lines=[f"id={u.id} tg={u.tg_id} bal={format_amount(u.balance)} vip={u.vip_tier} frozen={u.frozen}" for u in users]
            await update.message.reply_text("\n".join(lines) or "None")
        elif sub in ("freeze","unfreeze") and len(context.args)>=2:
            tg_id=int(context.args[1])
            u=session.execute(select(User).where(User.tg_id==tg_id)).scalar_one_or_none()
            if not u:
                await update.message.reply_text("User not found"); return
            u.frozen = (sub=="freeze")
            session.commit()
            await update.message.reply_text(f"User {'frozen' if u.frozen else 'unfrozen'}.")
        elif sub=="setedge" and len(context.args)>=3:
            game=context.args[1].lower()
            try: edge_val=float(context.args[2])
            except: await update.message.reply_text("Bad edge"); return
            if game in GAME_REGISTRY:
                GAME_REGISTRY[game].edge=edge_val
                Config.HOUSE_EDGES[game]=edge_val
                await update.message.reply_text(f"Edge updated for {game}")
            else:
                await update.message.reply_text("Game not found")
        elif sub=="setlimit" and len(context.args)>=3:
            typ=context.args[1]; val=float(context.args[2])
            if typ=="min": Config.MIN_BET=val
            elif typ=="max": Config.MAX_BET=val
            elif typ=="maxwin":
                Config.MAX_WIN=val
                for g in GAME_REGISTRY.values():
                    g.max_win=Decimal(str(val))
            else:
                await update.message.reply_text("Use min|max|maxwin"); return
            await update.message.reply_text("Limit updated.")
        elif sub=="rotate_seed_user" and len(context.args)>=2:
            tg_id=int(context.args[1])
            u=session.execute(select(User).where(User.tg_id==tg_id)).scalar_one_or_none()
            if not u: await update.message.reply_text("User not found"); return
            old_seed,new_hash=UserService.rotate_server_seed(session,u)
            await update.message.reply_text(f"Rotated. Old:{old_seed} New hash:{new_hash}")
        elif sub=="approve_withdraw" and len(context.args)>=3:
            wid=int(context.args[1]); txh=context.args[2]
            try:
                WalletService.admin_process_withdrawal(session,wid,True,update.effective_user.id,tx_hash=txh)
                await update.message.reply_text("Approved & paid.")
            except Exception as e:
                await update.message.reply_text(f"Error: {e}")
        elif sub=="deny_withdraw" and len(context.args)>=3:
            wid=int(context.args[1]); reason=" ".join(context.args[2:])
            try:
                WalletService.admin_process_withdrawal(session,wid,False,update.effective_user.id,reason=reason)
                await update.message.reply_text("Denied & refunded.")
            except Exception as e:
                await update.message.reply_text(f"Error: {e}")
        elif sub=="jackpot_add" and len(context.args)>=2:
            amt=Decimal(context.args[1])
            jp=session.query(SlotJackpot).first()
            if not jp:
                jp=SlotJackpot(pool=Decimal("0")); session.add(jp); session.flush()
            jp.pool += amt
            session.commit()
            await update.message.reply_text(f"Jackpot new pool {format_amount(jp.pool)}")
        elif sub=="kyc_flag" and len(context.args)>=2:
            tg_id=int(context.args[1]); u=session.execute(select(User).where(User.tg_id==tg_id)).scalar_one_or_none()
            if u: u.kyc_flag=True; session.commit(); await update.message.reply_text("Flagged")
            else: await update.message.reply_text("User not found")
        elif sub=="kyc_clear" and len(context.args)>=2:
            tg_id=int(context.args[1]); u=session.execute(select(User).where(User.tg_id==tg_id)).scalar_one_or_none()
            if u: u.kyc_flag=False; session.commit(); await update.message.reply_text("Cleared")
            else: await update.message.reply_text("User not found")
        elif sub=="regions_block" and len(context.args)>=2:
            regs=context.args[1].upper()
            existing=os.getenv("BLOCKED_REGIONS","")
            merged=existing + ("," if existing and regs else "") + regs
            os.environ["BLOCKED_REGIONS"]=merged
            await update.message.reply_text(f"Blocked regions: {merged}")
        elif sub=="regions_unblock" and len(context.args)>=2:
            remove={r.strip().upper() for r in context.args[1].split(",")}
            current={r.strip().upper() for r in os.getenv("BLOCKED_REGIONS","").split(",") if r.strip()}
            current-=remove
            new_val=",".join(sorted(current))
            os.environ["BLOCKED_REGIONS"]=new_val
            await update.message.reply_text(f"Blocked regions: {new_val}")
        elif sub=="pvp_list":
            matches=session.execute(select(PvPMatch).order_by(PvPMatch.id.desc()).limit(20)).scalars().all()
            lines=[f"#{m.id} {m.game} status={m.status} pot={format_amount(m.pot)} winner={m.winner_user_id}" for m in matches]
            await update.message.reply_text("\n".join(lines) or "None")
        elif sub=="cancel_match" and len(context.args)>=2:
            mid=int(context.args[1]); m=session.get(PvPMatch,mid)
            if not m: await update.message.reply_text("Not found"); return
            if m.status in ("completed","cancelled"):
                await update.message.reply_text("Already done"); return
            if m.creator_user_id:
                cu=session.get(User,m.creator_user_id); cu.balance += m.entry_fee
            if m.opponent_user_id and m.status=="active":
                ou=session.get(User,m.opponent_user_id); ou.balance += m.entry_fee
            m.status="cancelled"; session.commit()
            await update.message.reply_text("Cancelled.")
        elif sub=="session_list":
            sess=session.execute(select(GameSession).order_by(GameSession.id.desc()).limit(20)).scalars().all()
            lines=[f"#{s.id} user={s.user_id} game={s.game} active={s.active}" for s in sess]
            await update.message.reply_text("\n".join(lines) or "None")
        elif sub=="end_session" and len(context.args)>=2:
            sid=int(context.args[1]); gs=session.get(GameSession,sid)
            if gs: gs.active=False; session.commit(); await update.message.reply_text("Session ended.")
            else: await update.message.reply_text("Not found")
        elif sub=="exposure_snapshot":
            total_bal=session.execute(select(func.coalesce(func.sum(User.balance),0))).scalar_one()
            jp=session.query(SlotJackpot).first()
            snap=ExposureSnapshot(total_user_balances=total_bal, jackpot_pool=jp.pool if jp else Decimal("0"))
            session.add(snap); session.commit()
            await update.message.reply_text(f"Exposure snapshot {snap.id}")
        elif sub=="collusion_report":
            data=detect_collusion_patterns(session)
            await update.message.reply_text(json.dumps(data, indent=2)[:3800])
        else:
            await update.message.reply_text("Unknown admin cmd.\n"+ADMIN_HELP)

# Callback handler
async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.callback_query: return
    data_raw=update.callback_query.data
    payload=verify_signed_callback(data_raw)
    if not payload:
        await update.callback_query.answer("Invalid/expired", show_alert=True)
        return
    action=payload["a"]
    with SessionLocal() as session:
        user=UserService.get_or_create(session, update.effective_user)
        if action=="menu_games":
            await update.callback_query.edit_message_text("Games:\n"+", ".join(sorted(GAME_REGISTRY.keys())))
        elif action=="menu_deposit":
            await update.callback_query.edit_message_text("Use /deposit <chain>")
        elif action=="menu_withdraw":
            await update.callback_query.edit_message_text("Use /withdraw <chain> <address> <amount>")
        elif action=="menu_profile":
            await profile_cmd(update, context)
        elif action=="menu_vip":
            await update.callback_query.edit_message_text(f"VIP: {user.vip_tier} Rewards via /rewards")
        elif action=="menu_pf":
            await update.callback_query.edit_message_text("Provably Fair uses server_seed + client_seed + nonce => HMAC => result. /seed rotate to reveal old seed.")
        elif action=="menu_help":
            await update.callback_query.edit_message_text(HELP_TEXT[:3500])
# -------------------- CONTINUATION (Append to previously received final.py) --------------------
# Finish callback handler (claim_reward branch) and add remaining branches + background tasks, app build, main()

        elif action=="claim_reward":
            rid=payload["d"].get("rid")
            try:
                r=RewardService.claim_reward(session,user,rid)
                await update.callback_query.edit_message_text(
                    f"Claimed reward #{r.id} {format_amount(r.amount)}. Balance updated."
                )
            except Exception as e:
                await update.callback_query.answer(str(e), show_alert=True)
        else:
            await update.callback_query.answer("Unknown action", show_alert=True)
    try:
        await update.callback_query.answer()
    except:
        pass


# -------------------- BACKGROUND TASKS --------------------

async def deposit_monitor_loop():
    while True:
        try:
            with SessionLocal() as session:
                WalletService.credit_deposits(session)
        except Exception:
            logger.exception("deposit_monitor_loop error")
        await asyncio.sleep(10)

async def leaderboard_snapshot_loop():
    while True:
        try:
            with SessionLocal() as session:
                top = session.execute(
                    select(User.username, func.coalesce(func.sum(Bet.amount),0).label("wager"))
                    .join(Bet, Bet.user_id==User.id)
                    .group_by(User.id)
                    .order_by(func.sum(Bet.amount).desc())
                    .limit(10)
                ).all()
                lb = Leaderboard(period="rolling", metric="wagered",
                                 snapshot_json={"top":[(u, str(float(w))) for u,w in top]})
                session.add(lb); session.commit()
        except Exception:
            logger.exception("leaderboard_snapshot_loop error")
        await asyncio.sleep(3600)

async def rewards_accrual_loop():
    while True:
        try:
            with SessionLocal() as session:
                RewardService.accrue_periodic_rewards(session)
        except Exception:
            logger.exception("rewards_accrual_loop error")
        await asyncio.sleep(3600)

async def onchain_listener_loop():
    if not Web3:
        logger.info("web3 not installed; skipping onchain listener (will rely on simple deposit monitor).")
        return
    # Very primitive placeholder: just waits (real implementation would poll tx hashes / subscribe)
    while True:
        await asyncio.sleep(300)


# -------------------- APPLICATION BUILD --------------------

async def post_init(app: Application):
    # Core command list (trim if near API limits)
    commands = [
        BotCommand("start","Main menu"),
        BotCommand("help","Help"),
        BotCommand("balance","Show balance"),
        BotCommand("games","List games"),
        BotCommand("bet","Place a bet"),
        BotCommand("deposit","Deposit address"),
        BotCommand("withdraw","Withdraw request"),
        BotCommand("profile","Profile & stats"),
        BotCommand("seed","Show/rotate seed"),
        BotCommand("verify","Verify a bet"),
        BotCommand("rewards","Claim rewards"),
        BotCommand("bj_start","Blackjack start"),
        BotCommand("mines_start","Start Mines"),
        BotCommand("crash_bet","Join Crash"),
        BotCommand("crash_cashout","Crash cashout"),
        BotCommand("pvp_create","Create PvP"),
        BotCommand("holdem_create","Create Hold'em"),
        BotCommand("table_list","List Hold'em"),
        BotCommand("rtp_report","RTP report"),
        BotCommand("session_token","REST token"),
        BotCommand("selftest","Quick PF test"),
    ]
    try:
        await app.bot.set_my_commands(commands)
    except Exception:
        logger.exception("Failed setting commands")


def build_application() -> Application:
    rate_limiter = AIORateLimiter(max_retries=2)
    app = ApplicationBuilder().token(Config.BOT_TOKEN).rate_limiter(rate_limiter).post_init(post_init).build()

    # Core
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("balance", balance_cmd))
    app.add_handler(CommandHandler("games", games_cmd))
    app.add_handler(CommandHandler("bet", bet_cmd))
    app.add_handler(CommandHandler("seed", seed_cmd))
    app.add_handler(CommandHandler("verify", verify_cmd))
    app.add_handler(CommandHandler("deposit", deposit_cmd))
    app.add_handler(CommandHandler("withdraw", withdraw_cmd))
    app.add_handler(CommandHandler("profile", profile_cmd))
    app.add_handler(CommandHandler("tip", tip_cmd))
    app.add_handler(CommandHandler("rain", rain_cmd))
    app.add_handler(CommandHandler("rewards", rewards_cmd))
    app.add_handler(CommandHandler("selfexclude", selfexclude_cmd))
    app.add_handler(CommandHandler("cooldown", cooldown_cmd))
    # Blackjack
    app.add_handler(CommandHandler("bj_start", bj_start_cmd))
    app.add_handler(CommandHandler("bj_hit", bj_hit_cmd))
    app.add_handler(CommandHandler("bj_stand", bj_stand_cmd))
    app.add_handler(CommandHandler("bj_double", bj_double_cmd))
    app.add_handler(CommandHandler("bj_split", bj_split_cmd))
    # Mines interactive
    app.add_handler(CommandHandler("mines_start", mines_start_cmd))
    app.add_handler(CommandHandler("mines_pick", mines_pick_cmd))
    app.add_handler(CommandHandler("mines_cashout", mines_cashout_cmd))
    # Crash
    app.add_handler(CommandHandler("crash_bet", crash_bet_cmd))
    app.add_handler(CommandHandler("crash_cashout", crash_cashout_cmd))
    # PvP
    app.add_handler(CommandHandler("pvp_create", pvp_create_cmd))
    app.add_handler(CommandHandler("pvp_join", pvp_join_cmd))
    app.add_handler(CommandHandler("pvp_round", pvp_round_cmd))
    app.add_handler(CommandHandler("pvp_forfeit", pvp_forfeit_cmd))
    # Hold'em
    app.add_handler(CommandHandler("holdem_create", holdem_create_cmd))
    app.add_handler(CommandHandler("holdem_join", holdem_join_cmd))
    app.add_handler(CommandHandler("holdem_leave", holdem_leave_cmd))
    app.add_handler(CommandHandler("holdem_start", holdem_start_cmd))
    app.add_handler(CommandHandler("holdem_action", holdem_action_cmd))
    app.add_handler(CommandHandler("table_list", table_list_cmd))
    # Analytics / Misc
    app.add_handler(CommandHandler("rtp_report", rtp_report_cmd))
    app.add_handler(CommandHandler("session_token", session_token_cmd))
    app.add_handler(CommandHandler("fingerprint", fingerprint_cmd))
    app.add_handler(CommandHandler("audit_tail", audit_tail_cmd))
    app.add_handler(CommandHandler("exposure_snapshot", exposure_snapshot_cmd))
    app.add_handler(CommandHandler("collusion_report", collusion_report_cmd))
    app.add_handler(CommandHandler("kyc_flag", kyc_flag_cmd))
    # Simulations / tests
    app.add_handler(CommandHandler("simulate_dice_ev", simulate_dice_ev_cmd))
    app.add_handler(CommandHandler("simulate_limbo_ev", simulate_limbo_ev_cmd))
    app.add_handler(CommandHandler("selftest", selftest_cmd))
    app.add_handler(CommandHandler("selftest_property", selftest_property_cmd))
    # Admin
    app.add_handler(CommandHandler("admin", admin_cmd))
    # Callback
    app.add_handler(CallbackQueryHandler(callback_handler))
    return app


# -------------------- MAIN --------------------

def main():
    if Config.BOT_TOKEN == "REPLACE_WITH_REAL_TOKEN":
        logger.error("Set BOT_TOKEN env var before running.")
        return
    init_db()
    app = build_application()

    async def runner():
        # Background tasks
        asyncio.create_task(deposit_monitor_loop())
        asyncio.create_task(leaderboard_snapshot_loop())
        asyncio.create_task(rewards_accrual_loop())
        asyncio.create_task(crash_round_loop())
        asyncio.create_task(onchain_listener_loop())
        # REST API
        asyncio.create_task(start_rest_api())

        await app.initialize()
        await app.start()
        logger.info("Bot started.")
        await app.updater.start_polling(drop_pending_updates=True)
        await app.updater.idle()
        await app.stop()
        await app.shutdown()

    try:
        asyncio.run(runner())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down gracefully.")

if __name__ == "__main__":
    main()

# -------------------- END OF final.py --------------------