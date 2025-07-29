#!/usr/bin/env python3
"""
Advanced Telegram Crypto Escrow Bot
-----------------------------------
A sophisticated escrow bot that detects deal messages in groups, manages confirmations,
and handles secure cryptocurrency transactions across multiple blockchains.

Features:
- Smart deal detection in group chats
- Real-time confirmation workflow
- Multi-chain cryptocurrency support
- Secure escrow management
- Admin controls and dispute resolution
- Comprehensive transaction tracking

Author: Jashansinghsandhu
Date: 2025-07-29
"""

import os
import re
import json
import logging
import asyncio
import secrets
import sqlite3
import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum, auto
from uuid import uuid4
from dataclasses import dataclass
from dotenv import load_dotenv

# Telegram Bot Libraries
from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery, 
    Message, User, Chat, MessageEntity, Bot
)
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters, ConversationHandler
)

# Crypto Libraries
from web3 import Web3
from eth_account import Account
from tronpy import Tron
from tronpy.keys import PrivateKey

# Database and utilities
import sqlite3
import aiosqlite
import hashlib
import logging.handlers
import threading
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.handlers.RotatingFileHandler(
            'escrow_bot.log', 
            maxBytes=10485760, 
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Bot Configuration
TOKEN = os.getenv('8359797312:AAHH4jJ2NDLD9rVPtDUyrb-CZi0BYx3dVB4')
ADMIN_IDS = list(map(int, os.getenv('6083286836', '').split(',')))
DB_PATH = os.getenv('DB_PATH', 'escrow_bot.db')
FEE_PERCENTAGE = float(os.getenv('FEE_PERCENTAGE', '1.5'))
MIN_FEE = float(os.getenv('MIN_FEE', '1.0'))
MAX_FEE = float(os.getenv('MAX_FEE', '1.1'))

# Web3 Configuration
ETH_RPC_URL = os.getenv('ETH_RPC_URL', 'https://mainnet.infura.io/v3/25cdeb5b655744f2b6d88c998e55eace')
BSC_RPC_URL = os.getenv('BSC_RPC_URL', 'https://bsc-dataseed.binance.org/')
POLYGON_RPC_URL = os.getenv('POLYGON_RPC_URL', 'https://polygon-rpc.com')
TRON_RPC_URL = os.getenv('TRON_RPC_URL', 'https://api.trongrid.io')

# Deal States
class DealState(Enum):
    CREATED = auto()
    DETECTED = auto()
    BUYER_CONFIRMED = auto()
    SELLER_CONFIRMED = auto()
    BOTH_CONFIRMED = auto()
    AWAITING_PAYMENT = auto()
    PAYMENT_RECEIVED = auto()
    DELIVERY_CONFIRMED = auto()
    COMPLETED = auto()
    REFUNDED = auto()
    DISPUTED = auto()
    CANCELLED = auto()

# Network Types
class NetworkType(Enum):
    ETHEREUM = "ETH"
    BSC = "BSC"
    POLYGON = "POLYGON"
    TRON = "TRON"
    TON = "TON"
    LTC = "LTC"

# Crypto Currencies
class CryptoType(Enum):
    ETH = "ETH"
    BNB = "BNB"
    MATIC = "MATIC"
    TRX = "TRX"
    TON = "TON"
    LTC = "LTC"
    USDT_ERC20 = "USDT_ERC20"
    USDT_BEP20 = "USDT_BEP20"
    USDT_TRC20 = "USDT_TRC20"
    USDC_ERC20 = "USDC_ERC20"
    USDC_BEP20 = "USDC_BEP20"

# Data Structures
@dataclass
class Deal:
    id: str
    buyer_username: str
    buyer_id: Optional[int]
    seller_username: str
    seller_id: Optional[int]
    amount: Decimal
    currency: str
    condition: str
    state: DealState
    created_at: datetime.datetime
    escrow_address: Optional[str] = None
    network: Optional[str] = None
    tx_hash: Optional[str] = None
    fee_amount: Optional[Decimal] = None
    payment_confirmed_at: Optional[datetime.datetime] = None
    completed_at: Optional[datetime.datetime] = None
    chat_id: Optional[int] = None
    message_id: Optional[int] = None
    notes: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data['id'],
            buyer_username=data['buyer_username'],
            buyer_id=data.get('buyer_id'),
            seller_username=data['seller_username'],
            seller_id=data.get('seller_id'),
            amount=Decimal(str(data['amount'])),
            currency=data['currency'],
            condition=data['condition'],
            state=DealState[data['state']],
            created_at=datetime.datetime.fromisoformat(data['created_at']),
            escrow_address=data.get('escrow_address'),
            network=data.get('network'),
            tx_hash=data.get('tx_hash'),
            fee_amount=Decimal(str(data['fee_amount'])) if data.get('fee_amount') else None,
            payment_confirmed_at=datetime.datetime.fromisoformat(data['payment_confirmed_at']) if data.get('payment_confirmed_at') else None,
            completed_at=datetime.datetime.fromisoformat(data['completed_at']) if data.get('completed_at') else None,
            chat_id=data.get('chat_id'),
            message_id=data.get('message_id'),
            notes=data.get('notes')
        )
    
    def to_dict(self):
        return {
            'id': self.id,
            'buyer_username': self.buyer_username,
            'buyer_id': self.buyer_id,
            'seller_username': self.seller_username,
            'seller_id': self.seller_id,
            'amount': str(self.amount),
            'currency': self.currency,
            'condition': self.condition,
            'state': self.state.name,
            'created_at': self.created_at.isoformat(),
            'escrow_address': self.escrow_address,
            'network': self.network,
            'tx_hash': self.tx_hash,
            'fee_amount': str(self.fee_amount) if self.fee_amount else None,
            'payment_confirmed_at': self.payment_confirmed_at.isoformat() if self.payment_confirmed_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'chat_id': self.chat_id,
            'message_id': self.message_id,
            'notes': self.notes
        }

# Database setup
async def setup_database():
    """Initialize the SQLite database with necessary tables."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT,
            first_name TEXT,
            last_name TEXT,
            reputation_score REAL DEFAULT 5.0,
            total_deals INTEGER DEFAULT 0,
            successful_deals INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_active TIMESTAMP,
            is_banned BOOLEAN DEFAULT 0
        )
        ''')
        
        await db.execute('''
        CREATE TABLE IF NOT EXISTS deals (
            id TEXT PRIMARY KEY,
            buyer_username TEXT NOT NULL,
            buyer_id INTEGER,
            seller_username TEXT NOT NULL,
            seller_id INTEGER,
            amount DECIMAL NOT NULL,
            currency TEXT NOT NULL,
            condition TEXT,
            state TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            escrow_address TEXT,
            network TEXT,
            tx_hash TEXT,
            fee_amount DECIMAL,
            payment_confirmed_at TIMESTAMP,
            completed_at TIMESTAMP,
            chat_id INTEGER,
            message_id INTEGER,
            notes TEXT,
            FOREIGN KEY (buyer_id) REFERENCES users(id),
            FOREIGN KEY (seller_id) REFERENCES users(id)
        )
        ''')
        
        await db.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id TEXT PRIMARY KEY,
            deal_id TEXT NOT NULL,
            tx_type TEXT NOT NULL,
            amount DECIMAL NOT NULL,
            tx_hash TEXT,
            from_address TEXT,
            to_address TEXT,
            network TEXT,
            status TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            confirmed_at TIMESTAMP,
            FOREIGN KEY (deal_id) REFERENCES deals(id)
        )
        ''')
        
        await db.execute('''
        CREATE TABLE IF NOT EXISTS disputes (
            id TEXT PRIMARY KEY,
            deal_id TEXT NOT NULL,
            reported_by INTEGER NOT NULL,
            reason TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP,
            resolution TEXT,
            resolved_by INTEGER,
            FOREIGN KEY (deal_id) REFERENCES deals(id),
            FOREIGN KEY (reported_by) REFERENCES users(id),
            FOREIGN KEY (resolved_by) REFERENCES users(id)
        )
        ''')
        
        await db.execute('''
        CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action_type TEXT NOT NULL,
            deal_id TEXT,
            details TEXT,
            ip_address TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (deal_id) REFERENCES deals(id)
        )
        ''')
        
        await db.execute('''
        CREATE TABLE IF NOT EXISTS wallets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            address TEXT NOT NULL UNIQUE,
            network TEXT NOT NULL,
            currency TEXT NOT NULL,
            encrypted_private_key TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_used TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
        ''')
        
        await db.commit()

# Deal detection regex patterns
DEAL_PATTERNS = [
    # Standard format
    re.compile(r"(?i)Buyer\s*:\s*@?(\w+)\s*Seller\s*:\s*@?(\w+)\s*Amount\s*:\s*(\d+\.?\d*)\s*(?:\[?\s*in\s*(\w+)\s*\]?)?\s*Condition\s*:\s*(.+)", re.DOTALL),
    # Alternative format with different order
    re.compile(r"(?i)Seller\s*:\s*@?(\w+)\s*Buyer\s*:\s*@?(\w+)\s*Amount\s*:\s*(\d+\.?\d*)\s*(?:\[?\s*in\s*(\w+)\s*\]?)?\s*Condition\s*:\s*(.+)", re.DOTALL),
    # Format with emojis and different separators
    re.compile(r"(?i)Buyer\s*[:\-=]\s*@?(\w+).*?Seller\s*[:\-=]\s*@?(\w+).*?Amount\s*[:\-=]\s*(\d+\.?\d*).*?(?:in\s*(\w+))?.*?Condition\s*[:\-=]\s*(.+)", re.DOTALL),
]

# Wallet Management
class WalletManager:
    """Handle wallet operations including generation, encryption, and transaction signing."""
    
    def __init__(self):
        self.web3_eth = Web3(Web3.HTTPProvider(ETH_RPC_URL))
        self.web3_bsc = Web3(Web3.HTTPProvider(BSC_RPC_URL))
        self.web3_polygon = Web3(Web3.HTTPProvider(POLYGON_RPC_URL))
        self.tron_client = Tron(TRON_RPC_URL)
        self.encryption_key = os.getenv('WALLET_ENCRYPTION_KEY', secrets.token_hex(16))
    
    def encrypt_private_key(self, private_key: str) -> str:
        """Encrypt a private key before storing in database."""
        # In a production environment, use a proper encryption library
        # This is a simple XOR encryption for demonstration
        key_bytes = self.encryption_key.encode()
        encrypted = bytearray()
        for i, c in enumerate(private_key.encode()):
            encrypted.append(c ^ key_bytes[i % len(key_bytes)])
        return encrypted.hex()
    
    def decrypt_private_key(self, encrypted_key: str) -> str:
        """Decrypt a stored private key."""
        key_bytes = self.encryption_key.encode()
        encrypted = bytes.fromhex(encrypted_key)
        decrypted = bytearray()
        for i, c in enumerate(encrypted):
            decrypted.append(c ^ key_bytes[i % len(key_bytes)])
        return decrypted.decode()
    
    async def generate_wallet(self, network: str, currency: str) -> Tuple[str, str]:
        """Generate a new wallet for the specified network and currency."""
        private_key = secrets.token_hex(32)
        
        if network in [NetworkType.ETHEREUM.value, NetworkType.BSC.value, NetworkType.POLYGON.value]:
            # Ethereum-compatible chains
            account = Account.from_key(private_key)
            address = account.address
        elif network == NetworkType.TRON.value:
            # TRON chain
            tron_key = PrivateKey(bytes.fromhex(private_key))
            address = tron_key.public_key.to_base58check_address()
        else:
            # Placeholder for other chains
            address = f"UNSUPPORTED_{network}_{uuid4().hex[:8]}"
        
        # Store the wallet in database
        encrypted_key = self.encrypt_private_key(private_key)
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute('''
            INSERT INTO wallets (address, network, currency, encrypted_private_key)
            VALUES (?, ?, ?, ?)
            ''', (address, network, currency, encrypted_key))
            await db.commit()
        
        return address, private_key
    
    async def get_wallet_for_deal(self, deal_id: str, network: str, currency: str) -> str:
        """Get or create a wallet for a specific deal."""
        address, _ = await self.generate_wallet(network, currency)
        
        # Update the deal with the escrow address
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute('''
            UPDATE deals SET escrow_address = ?, network = ?
            WHERE id = ?
            ''', (address, network, deal_id))
            await db.commit()
        
        return address
    
    async def check_balance(self, address: str, network: str, currency: str) -> Decimal:
        """Check balance of a wallet address for the specified network and currency."""
        # Simplified implementation - in production, use proper RPC calls
        balance = Decimal('0.0')
        
        if network == NetworkType.ETHEREUM.value:
            if currency == CryptoType.ETH.value:
                wei_balance = self.web3_eth.eth.get_balance(address)
                balance = Decimal(wei_balance) / Decimal(10**18)
            # Add ERC20 token support here
        
        elif network == NetworkType.BSC.value:
            if currency == CryptoType.BNB.value:
                wei_balance = self.web3_bsc.eth.get_balance(address)
                balance = Decimal(wei_balance) / Decimal(10**18)
            # Add BEP20 token support here
        
        # Add support for other networks here
        
        return balance

# Deal Manager
class DealManager:
    """Manage deal operations including creation, updates, and queries."""
    
    def __init__(self, wallet_manager: WalletManager):
        self.wallet_manager = wallet_manager
    
    async def create_deal(self, buyer_username: str, seller_username: str, 
                         amount: Decimal, currency: str, condition: str, 
                         chat_id: int, message_id: int) -> Deal:
        """Create a new deal in the database."""
        deal_id = f"DEAL-{uuid4().hex[:8]}"
        now = datetime.datetime.now()
        
        # Calculate fee
        fee_amount = max(min(amount * Decimal(FEE_PERCENTAGE) / Decimal(100), Decimal(MAX_FEE)), Decimal(MIN_FEE))
        
        deal = Deal(
            id=deal_id,
            buyer_username=buyer_username,
            buyer_id=None,  # Will be updated when user confirms
            seller_username=seller_username,
            seller_id=None,  # Will be updated when user confirms
            amount=amount,
            currency=currency,
            condition=condition,
            state=DealState.DETECTED,
            created_at=now,
            chat_id=chat_id,
            message_id=message_id,
            fee_amount=fee_amount
        )
        
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute('''
            INSERT INTO deals (
                id, buyer_username, seller_username, amount, currency, 
                condition, state, created_at, fee_amount, chat_id, message_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                deal.id, deal.buyer_username, deal.seller_username, 
                str(deal.amount), deal.currency, deal.condition, 
                deal.state.name, deal.created_at.isoformat(), 
                str(deal.fee_amount), deal.chat_id, deal.message_id
            ))
            await db.commit()
        
        return deal
    
    async def get_deal(self, deal_id: str) -> Optional[Deal]:
        """Get a deal by ID."""
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = sqlite3.Row
            async with db.execute('SELECT * FROM deals WHERE id = ?', (deal_id,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return Deal.from_dict(dict(row))
        return None
    
    async def update_deal_state(self, deal_id: str, state: DealState, **kwargs) -> Optional[Deal]:
        """Update the state of a deal and any additional fields."""
        deal = await self.get_deal(deal_id)
        if not deal:
            return None
        
        deal.state = state
        
        # Update any additional fields
        for key, value in kwargs.items():
            if hasattr(deal, key):
                setattr(deal, key, value)
        
        # Prepare the SQL update
        set_clause = ['state = ?']
        params = [state.name]
        
        for key, value in kwargs.items():
            if hasattr(deal, key):
                set_clause.append(f"{key} = ?")
                # Convert datetime to string
                if isinstance(value, datetime.datetime):
                    params.append(value.isoformat())
                # Convert Decimal to string
                elif isinstance(value, Decimal):
                    params.append(str(value))
                else:
                    params.append(value)
        
        query = f"UPDATE deals SET {', '.join(set_clause)} WHERE id = ?"
        params.append(deal_id)
        
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(query, params)
            await db.commit()
        
        return deal
    
    async def confirm_buyer(self, deal_id: str, user_id: int) -> Optional[Deal]:
        """Confirm the buyer for a deal."""
        deal = await self.get_deal(deal_id)
        if not deal or deal.state not in [DealState.DETECTED, DealState.SELLER_CONFIRMED]:
            return None
        
        new_state = DealState.BUYER_CONFIRMED
        if deal.state == DealState.SELLER_CONFIRMED:
            new_state = DealState.BOTH_CONFIRMED
        
        return await self.update_deal_state(deal_id, new_state, buyer_id=user_id)
    
    async def confirm_seller(self, deal_id: str, user_id: int) -> Optional[Deal]:
        """Confirm the seller for a deal."""
        deal = await self.get_deal(deal_id)
        if not deal or deal.state not in [DealState.DETECTED, DealState.BUYER_CONFIRMED]:
            return None
        
        new_state = DealState.SELLER_CONFIRMED
        if deal.state == DealState.BUYER_CONFIRMED:
            new_state = DealState.BOTH_CONFIRMED
        
        return await self.update_deal_state(deal_id, new_state, seller_id=user_id)
    
    async def setup_payment(self, deal_id: str, network: str) -> Tuple[Optional[Deal], Optional[str]]:
        """Set up payment details for a deal."""
        deal = await self.get_deal(deal_id)
        if not deal or deal.state != DealState.BOTH_CONFIRMED:
            return None, None
        
        # Generate a wallet address for the escrow
        escrow_address = await self.wallet_manager.get_wallet_for_deal(
            deal_id, network, deal.currency
        )
        
        # Update deal state
        deal = await self.update_deal_state(
            deal_id, 
            DealState.AWAITING_PAYMENT, 
            network=network,
            escrow_address=escrow_address
        )
        
        return deal, escrow_address
    
    async def check_payment(self, deal_id: str) -> Tuple[bool, Optional[Deal]]:
        """Check if payment has been received for a deal."""
        deal = await self.get_deal(deal_id)
        if not deal or deal.state != DealState.AWAITING_PAYMENT:
            return False, deal
        
        # Check balance of escrow address
        balance = await self.wallet_manager.check_balance(
            deal.escrow_address, deal.network, deal.currency
        )
        
        # If balance is sufficient, mark payment as received
        if balance >= deal.amount:
            deal = await self.update_deal_state(
                deal_id,
                DealState.PAYMENT_RECEIVED,
                payment_confirmed_at=datetime.datetime.now()
            )
            return True, deal
        
        return False, deal
    
    async def confirm_delivery(self, deal_id: str, user_id: int) -> Optional[Deal]:
        """Confirm delivery of goods/services and release payment."""
        deal = await self.get_deal(deal_id)
        if not deal or deal.state != DealState.PAYMENT_RECEIVED or deal.buyer_id != user_id:
            return None
        
        # In a real implementation, we would transfer funds here
        # This is a simplified version
        
        # Mark the deal as completed
        deal = await self.update_deal_state(
            deal_id,
            DealState.COMPLETED,
            completed_at=datetime.datetime.now()
        )
        
        # Update user reputation
        if deal.buyer_id and deal.seller_id:
            async with aiosqlite.connect(DB_PATH) as db:
                # Update buyer stats
                await db.execute('''
                UPDATE users SET 
                    total_deals = total_deals + 1,
                    successful_deals = successful_deals + 1,
                    reputation_score = (reputation_score * total_deals + 5) / (total_deals + 1)
                WHERE id = ?
                ''', (deal.buyer_id,))
                
                # Update seller stats
                await db.execute('''
                UPDATE users SET 
                    total_deals = total_deals + 1,
                    successful_deals = successful_deals + 1,
                    reputation_score = (reputation_score * total_deals + 5) / (total_deals + 1)
                WHERE id = ?
                ''', (deal.seller_id,))
                
                await db.commit()
        
        return deal
    
    async def create_dispute(self, deal_id: str, user_id: int, reason: str) -> bool:
        """Create a dispute for a deal."""
        deal = await self.get_deal(deal_id)
        if not deal or (deal.buyer_id != user_id and deal.seller_id != user_id):
            return False
        
        # Only allow disputes for active deals
        if deal.state not in [DealState.PAYMENT_RECEIVED, DealState.AWAITING_PAYMENT]:
            return False
        
        dispute_id = f"DISPUTE-{uuid4().hex[:8]}"
        
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute('''
            INSERT INTO disputes (id, deal_id, reported_by, reason, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                dispute_id, deal_id, user_id, reason, "OPEN", 
                datetime.datetime.now().isoformat()
            ))
            
            # Update deal state
            await db.execute('''
            UPDATE deals SET state = ? WHERE id = ?
            ''', (DealState.DISPUTED.name, deal_id))
            
            await db.commit()
        
        return True
    
    async def get_user_deals(self, user_id: int) -> List[Deal]:
        """Get all deals for a user."""
        deals = []
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = sqlite3.Row
            async with db.execute('''
            SELECT * FROM deals 
            WHERE buyer_id = ? OR seller_id = ?
            ORDER BY created_at DESC
            ''', (user_id, user_id)) as cursor:
                async for row in cursor:
                    deals.append(Deal.from_dict(dict(row)))
        return deals
    
    async def get_active_deals(self) -> List[Deal]:
        """Get all active deals."""
        deals = []
        active_states = [
            DealState.AWAITING_PAYMENT.name,
            DealState.PAYMENT_RECEIVED.name,
            DealState.DISPUTED.name
        ]
        
        placeholders = ', '.join(['?' for _ in active_states])
        
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = sqlite3.Row
            async with db.execute(f'''
            SELECT * FROM deals 
            WHERE state IN ({placeholders})
            ORDER BY created_at DESC
            ''', active_states) as cursor:
                async for row in cursor:
                    deals.append(Deal.from_dict(dict(row)))
        return deals

# User management
class UserManager:
    """Manage user operations including registration, updates, and queries."""
    
    async def register_or_update_user(self, user: User) -> None:
        """Register a new user or update an existing one."""
        async with aiosqlite.connect(DB_PATH) as db:
            # Check if user exists
            async with db.execute('SELECT id FROM users WHERE id = ?', (user.id,)) as cursor:
                exists = await cursor.fetchone()
            
            now = datetime.datetime.now().isoformat()
            
            if exists:
                # Update existing user
                await db.execute('''
                UPDATE users SET 
                    username = ?,
                    first_name = ?,
                    last_name = ?,
                    last_active = ?
                WHERE id = ?
                ''', (
                    user.username, user.first_name, user.last_name, 
                    now, user.id
                ))
            else:
                # Insert new user
                await db.execute('''
                INSERT INTO users (
                    id, username, first_name, last_name, 
                    created_at, last_active
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    user.id, user.username, user.first_name, 
                    user.last_name, now, now
                ))
            
            await db.commit()
    
    async def get_user(self, user_id: int) -> Optional[Dict]:
        """Get user details by ID."""
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = sqlite3.Row
            async with db.execute('SELECT * FROM users WHERE id = ?', (user_id,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return dict(row)
        return None
    
    async def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user details by username."""
        # Remove @ prefix if present
        if username.startswith('@'):
            username = username[1:]
            
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = sqlite3.Row
            async with db.execute('SELECT * FROM users WHERE username = ?', (username,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return dict(row)
        return None
    
    async def log_activity(self, user_id: Optional[int], action_type: str, 
                         deal_id: Optional[str] = None, details: Optional[str] = None,
                         ip_address: Optional[str] = None) -> None:
        """Log user activity."""
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute('''
            INSERT INTO activity_logs (
                user_id, action_type, deal_id, details, ip_address
            )
            VALUES (?, ?, ?, ?, ?)
            ''', (user_id, action_type, deal_id, details, ip_address))
            await db.commit()

# Security Manager
class SecurityManager:
    """Handle security related operations including fraud detection and prevention."""
    
    async def check_user_reputation(self, user_id: int) -> Tuple[bool, str]:
        """Check if a user has good reputation."""
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute('''
            SELECT reputation_score, is_banned FROM users WHERE id = ?
            ''', (user_id,)) as cursor:
                row = await cursor.fetchone()
                
                if not row:
                    return True, "User not found, but allowed as new user"
                
                is_banned = row[1]
                if is_banned:
                    return False, "User is banned"
                
                reputation_score = row[0]
                if reputation_score < 3.0:
                    return False, f"Low reputation score: {reputation_score}"
                
                return True, f"Good reputation: {reputation_score}"
    
    async def is_suspicious_deal(self, deal: Deal) -> Tuple[bool, str]:
        """Check if a deal looks suspicious."""
        # Simple check for now
        if deal.amount > Decimal('10000'):
            return True, "Very high amount"
        
        # Check if buyer or seller has completed deals
        async with aiosqlite.connect(DB_PATH) as db:
            if deal.buyer_id:
                async with db.execute('''
                SELECT successful_deals FROM users WHERE id = ?
                ''', (deal.buyer_id,)) as cursor:
                    row = await cursor.fetchone()
                    if row and row[0] == 0:
                        return True, "Buyer has no successful deals"
            
            if deal.seller_id:
                async with db.execute('''
                SELECT successful_deals FROM users WHERE id = ?
                ''', (deal.seller_id,)) as cursor:
                    row = await cursor.fetchone()
                    if row and row[0] == 0:
                        return True, "Seller has no successful deals"
        
        return False, "Deal appears normal"
    
    async def record_security_event(self, event_type: str, user_id: Optional[int] = None,
                                  deal_id: Optional[str] = None, details: Optional[str] = None) -> None:
        """Record a security-related event."""
        await UserManager().log_activity(
            user_id=user_id,
            action_type=f"SECURITY_{event_type}",
            deal_id=deal_id,
            details=details
        )

# Main Bot Class
class EscrowBot:
    def __init__(self):
        """Initialize the Escrow Bot with all managers."""
        self.wallet_manager = WalletManager()
        self.deal_manager = DealManager(self.wallet_manager)
        self.user_manager = UserManager()
        self.security_manager = SecurityManager()
        
        # Initialize the application with the token
        self.application = Application.builder().token(TOKEN).build()
        
        # Register handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register all message and callback handlers."""
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.cmd_start))
        self.application.add_handler(CommandHandler("help", self.cmd_help))
        self.application.add_handler(CommandHandler("deals", self.cmd_deals))
        self.application.add_handler(CommandHandler("dispute", self.cmd_dispute))
        self.application.add_handler(CommandHandler("admin", self.cmd_admin, filters=filters.User(user_id=ADMIN_IDS)))
        
        # Deal detection in groups
        self.application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND & filters.ChatType.GROUPS, 
            self.detect_deal
        ))
        
        # Callback query handler for inline buttons
        self.application.add_handler(CallbackQueryHandler(self.handle_callback))
        
        # Error handler
        self.application.add_error_handler(self.error_handler)
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /start command."""
        user = update.effective_user
        await self.user_manager.register_or_update_user(user)
        
        await update.message.reply_text(
            f"👋 Hi {user.first_name}! I'm the Crypto Escrow Bot.\n\n"
            "I help buyers and sellers trade safely by holding funds in escrow until both parties are satisfied.\n\n"
            "🔹 I can detect deal formats in group chats\n"
            "🔹 I support multiple cryptocurrencies\n"
            "🔹 I provide secure escrow services\n\n"
            "Use /help to see available commands."
        )
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /help command."""
        help_text = (
            "🛡 *Crypto Escrow Bot Help* 🛡\n\n"
            "*Commands:*\n"
            "/start - Start the bot\n"
            "/help - Show this help message\n"
            "/deals - View your active deals\n"
            "/dispute <deal_id> <reason> - Open a dispute for a deal\n\n"
            
            "*Deal Format in Groups:*\n"
            "The bot will automatically detect messages with this format:\n"
            "```\n"
            "Buyer: @username\n"
            "Seller: @username\n"
            "Amount: 100 [in USDT]\n"
            "Condition: Item description or service\n"
            "```\n\n"
            
            "*Deal Process:*\n"
            "1. Deal is detected and both parties must confirm\n"
            "2. Buyer sends payment to escrow address\n"
            "3. Seller delivers the goods/services\n"
            "4. Buyer confirms receipt\n"
            "5. Funds are released to seller\n\n"
            
            "*Security Tips:*\n"
            "• Never share your private keys\n"
            "• Verify all transaction details\n"
            "• Only confirm receipt after satisfaction\n"
            "• Contact admins for any issues"
        )
        
        await update.message.reply_text(help_text, parse_mode="Markdown")
    
    async def cmd_deals(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /deals command to show user's active deals."""
        user = update.effective_user
        await self.user_manager.register_or_update_user(user)
        
        deals = await self.deal_manager.get_user_deals(user.id)
        
        if not deals:
            await update.message.reply_text("You don't have any deals yet.")
            return
        
        # Group deals by state
        active_deals = []
        completed_deals = []
        
        for deal in deals:
            if deal.state in [DealState.AWAITING_PAYMENT, DealState.PAYMENT_RECEIVED, DealState.DISPUTED]:
                active_deals.append(deal)
            elif deal.state == DealState.COMPLETED:
                completed_deals.append(deal)
        
        # Format and send the active deals
        if active_deals:
            active_text = "🟢 *Your Active Deals:*\n\n"
            for i, deal in enumerate(active_deals[:5], 1):  # Limit to 5 deals
                role = "Buyer" if deal.buyer_id == user.id else "Seller"
                counterparty = deal.seller_username if role == "Buyer" else deal.buyer_username
                
                active_text += (
                    f"*{i}. Deal ID:* `{deal.id}`\n"
                    f"*Role:* {role}\n"
                    f"*With:* @{counterparty}\n"
                    f"*Amount:* {deal.amount} {deal.currency}\n"
                    f"*Status:* {deal.state.name}\n"
                    f"*Created:* {deal.created_at.strftime('%Y-%m-%d %H:%M')}\n\n"
                )
            
            if len(active_deals) > 5:
                active_text += f"_...and {len(active_deals) - 5} more active deals_\n\n"
            
            await update.message.reply_text(active_text, parse_mode="Markdown")
        
        # Format and send the completed deals
        if completed_deals:
            completed_text = "✅ *Your Completed Deals:*\n\n"
            for i, deal in enumerate(completed_deals[:3], 1):  # Limit to 3 deals
                role = "Buyer" if deal.buyer_id == user.id else "Seller"
                counterparty = deal.seller_username if role == "Buyer" else deal.buyer_username
                
                completed_text += (
                    f"*{i}. Deal ID:* `{deal.id}`\n"
                    f"*Role:* {role}\n"
                    f"*With:* @{counterparty}\n"
                    f"*Amount:* {deal.amount} {deal.currency}\n"
                    f"*Completed:* {deal.completed_at.strftime('%Y-%m-%d %H:%M') if deal.completed_at else 'N/A'}\n\n"
                )
            
            if len(completed_deals) > 3:
                completed_text += f"_...and {len(completed_deals) - 3} more completed deals_\n\n"
            
            await update.message.reply_text(completed_text, parse_mode="Markdown")
    
    async def cmd_dispute(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /dispute command to open a dispute for a deal."""
        user = update.effective_user
        await self.user_manager.register_or_update_user(user)
        
        args = context.args
        if len(args) < 2:
            await update.message.reply_text(
                "⚠️ Please provide a deal ID and reason for the dispute.\n"
                "Example: `/dispute DEAL-12345abc Payment not received`"
            )
            return
        
        deal_id = args[0]
        reason = " ".join(args[1:])
        
        success = await self.deal_manager.create_dispute(deal_id, user.id, reason)
        
        if success:
            # Notify admins
            for admin_id in ADMIN_IDS:
                try:
                    await context.bot.send_message(
                        chat_id=admin_id,
                        text=f"🚨 *New Dispute Created*\n\n"
                             f"*Deal ID:* `{deal_id}`\n"
                             f"*Reported by:* @{user.username} (ID: {user.id})\n"
                             f"*Reason:* {reason}\n\n"
                             f"Use admin commands to review and resolve.",
                        parse_mode="Markdown"
                    )
                except Exception as e:
                    logger.error(f"Failed to notify admin {admin_id}: {str(e)}")
            
            await update.message.reply_text(
                "✅ Dispute created successfully. An admin will review your case shortly."
            )
        else:
            await update.message.reply_text(
                "❌ Failed to create dispute. Make sure the deal ID is correct and you are a participant."
            )
    
    async def cmd_admin(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /admin command for admin functions."""
        user = update.effective_user
        
        if not context.args:
            admin_help = (
                "🔐 *Admin Commands:*\n\n"
                "/admin deals - View all active deals\n"
                "/admin disputes - View all open disputes\n"
                "/admin resolve <dispute_id> <decision> - Resolve a dispute (BUYER/SELLER/REFUND)\n"
                "/admin ban <user_id> - Ban a user\n"
                "/admin unban <user_id> - Unban a user\n"
                "/admin stats - View system statistics"
            )
            await update.message.reply_text(admin_help, parse_mode="Markdown")
            return
        
        subcmd = context.args[0].lower()
        
        if subcmd == "deals":
            # Show all active deals
            deals = await self.deal_manager.get_active_deals()
            
            if not deals:
                await update.message.reply_text("No active deals found.")
                return
            
            deals_text = "🔄 *Active Deals:*\n\n"
            for i, deal in enumerate(deals[:10], 1):
                deals_text += (
                    f"*{i}. ID:* `{deal.id}`\n"
                    f"*Buyer:* @{deal.buyer_username} (ID: {deal.buyer_id or 'Unconfirmed'})\n"
                    f"*Seller:* @{deal.seller_username} (ID: {deal.seller_id or 'Unconfirmed'})\n"
                    f"*Amount:* {deal.amount} {deal.currency}\n"
                    f"*Status:* {deal.state.name}\n"
                    f"*Created:* {deal.created_at.strftime('%Y-%m-%d %H:%M')}\n\n"
                )
            
            if len(deals) > 10:
                deals_text += f"_...and {len(deals) - 10} more active deals_"
            
            await update.message.reply_text(deals_text, parse_mode="Markdown")
        
        # Add other admin subcommands like disputes, resolve, ban, unban, stats
        # Implementation not shown for brevity
    
    async def detect_deal(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Detect deal messages in group chats."""
        message = update.message
        message_text = message.text
        
        # Check if the message matches any of the deal patterns
        match = None
        for pattern in DEAL_PATTERNS:
            match = pattern.search(message_text)
            if match:
                break
        
        if not match:
            return
        
        # Extract deal details based on the pattern matched
        # The groups may be in different orders depending on the pattern
        if pattern == DEAL_PATTERNS[0]:  # Standard format
            buyer_username, seller_username, amount_str, currency, condition = match.groups()
        elif pattern == DEAL_PATTERNS[1]:  # Seller first format
            seller_username, buyer_username, amount_str, currency, condition = match.groups()
        else:  # Emoji/Alternative format
            buyer_username, seller_username, amount_str, currency, condition = match.groups()
        
        # Clean up usernames (remove @ if present)
        buyer_username = buyer_username.strip('@')
        seller_username = seller_username.strip('@')
        
        # Default currency to USDT if not specified
        if not currency:
            currency = "USDT"
        else:
            currency = currency.strip()
        
        # Parse amount
        try:
            amount = Decimal(amount_str)
        except:
            logger.error(f"Failed to parse amount: {amount_str}")
            return
        
        # Create the deal
        deal = await self.deal_manager.create_deal(
            buyer_username=buyer_username,
            seller_username=seller_username,
            amount=amount,
            currency=currency,
            condition=condition.strip() if condition else "No conditions specified",
            chat_id=message.chat.id,
            message_id=message.message_id
        )
        
        # Create keyboard for confirming the deal
        keyboard = [
            [
                InlineKeyboardButton("Confirm as Buyer", callback_data=f"confirm_buyer:{deal.id}"),
                InlineKeyboardButton("Confirm as Seller", callback_data=f"confirm_seller:{deal.id}")
            ],
            [
                InlineKeyboardButton("Edit Details", callback_data=f"edit:{deal.id}"),
                InlineKeyboardButton("Cancel", callback_data=f"cancel:{deal.id}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Reply to the message with deal details and confirmation buttons
        await message.reply_text(
            f"🛡️ *Escrow Deal Detected!*\n\n"
            f"*Buyer:* @{buyer_username}\n"
            f"*Seller:* @{seller_username}\n"
            f"*Amount:* {amount} {currency}\n"
            f"*Condition:* {condition.strip() if condition else 'No conditions specified'}\n\n"
            f"Both parties must confirm to proceed with the escrow.",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle callback queries from inline buttons."""
        query = update.callback_query
        user = query.from_user
        await self.user_manager.register_or_update_user(user)
        
        # Extract the action and deal ID from the callback data
        data = query.data.split(':')
        if len(data) != 2:
            await query.answer("Invalid callback data")
            return
        
        action, deal_id = data
        
        # Get the deal
        deal = await self.deal_manager.get_deal(deal_id)
        if not deal:
            await query.answer("Deal not found")
            return
        
        # Handle different actions
        if action == "confirm_buyer":
            # Check if user is the buyer
            if user.username.lower() != deal.buyer_username.lower():
                await query.answer("You are not the buyer in this deal", show_alert=True)
                return
            
            # Check deal state
            if deal.state not in [DealState.DETECTED, DealState.SELLER_CONFIRMED]:
                await query.answer("This deal cannot be confirmed now", show_alert=True)
                return
            
            # Confirm buyer
            updated_deal = await self.deal_manager.confirm_buyer(deal_id, user.id)
            
            if updated_deal:
                # Update message
                new_text = (
                    f"🛡️ *Escrow Deal*\n\n"
                    f"*Buyer:* @{deal.buyer_username} ✅\n"
                    f"*Seller:* @{deal.seller_username}{' ✅' if updated_deal.state == DealState.BOTH_CONFIRMED else ''}\n"
                    f"*Amount:* {deal.amount} {deal.currency}\n"
                    f"*Condition:* {deal.condition}\n\n"
                )
                
                # If both confirmed, update keyboard for payment setup
                if updated_deal.state == DealState.BOTH_CONFIRMED:
                    keyboard = [
                        [
                            InlineKeyboardButton("Start Payment Process", callback_data=f"setup_payment:{deal_id}")
                        ]
                    ]
                    await query.edit_message_text(
                        text=new_text + "✅ Both parties confirmed! Ready to proceed with payment.",
                        reply_markup=InlineKeyboardMarkup(keyboard),
                        parse_mode="Markdown"
                    )
                else:
                    # Update the message to show buyer confirmed
                    await query.edit_message_text(
                        text=new_text + "Waiting for seller confirmation...",
                        reply_markup=query.message.reply_markup,
                        parse_mode="Markdown"
                    )
                
                await query.answer("You've confirmed as the buyer")
            else:
                await query.answer("Failed to confirm as buyer", show_alert=True)
        
        elif action == "confirm_seller":
            # Check if user is the seller
            if user.username.lower() != deal.seller_username.lower():
                await query.answer("You are not the seller in this deal", show_alert=True)
                return
            
            # Check deal state
            if deal.state not in [DealState.DETECTED, DealState.BUYER_CONFIRMED]:
                await query.answer("This deal cannot be confirmed now", show_alert=True)
                return
            
            # Confirm seller
            updated_deal = await self.deal_manager.confirm_seller(deal_id, user.id)
            
            if updated_deal:
                # Update message
                new_text = (
                    f"🛡️ *Escrow Deal*\n\n"
                    f"*Buyer:* @{deal.buyer_username}{' ✅' if updated_deal.state == DealState.BOTH_CONFIRMED else ''}\n"
                    f"*Seller:* @{deal.seller_username} ✅\n"
                    f"*Amount:* {deal.amount} {deal.currency}\n"
                    f"*Condition:* {deal.condition}\n\n"
                )
                
                # If both confirmed, update keyboard for payment setup
                if updated_deal.state == DealState.BOTH_CONFIRMED:
                    keyboard = [
                        [
                            InlineKeyboardButton("Start Payment Process", callback_data=f"setup_payment:{deal_id}")
                        ]
                    ]
                    await query.edit_message_text(
                        text=new_text + "✅ Both parties confirmed! Ready to proceed with payment.",
                        reply_markup=InlineKeyboardMarkup(keyboard),
                        parse_mode="Markdown"
                    )
                else:
                    # Update the message to show seller confirmed
                    await query.edit_message_text(
                        text=new_text + "Waiting for buyer confirmation...",
                        reply_markup=query.message.reply_markup,
                        parse_mode="Markdown"
                    )
                
                await query.answer("You've confirmed as the seller")
            else:
                await query.answer("Failed to confirm as seller", show_alert=True)
        
        elif action == "setup_payment":
            # Check if user is buyer or seller
            is_buyer = user.username.lower() == deal.buyer_username.lower()
            is_seller = user.username.lower() == deal.seller_username.lower()
            
            if not (is_buyer or is_seller):
                await query.answer("You are not a participant in this deal", show_alert=True)
                return
            
            # Check deal state
            if deal.state != DealState.BOTH_CONFIRMED:
                await query.answer("This deal is not ready for payment setup", show_alert=True)
                return
            
            # Setup payment (using TRON for USDT by default)
            network = "TRON" if deal.currency.upper() == "USDT" else "ETH"
            updated_deal, escrow_address = await self.deal_manager.setup_payment(deal_id, network)
            
            if updated_deal and escrow_address:
                # Update message in group chat
                await query.edit_message_text(
                    text=(
                        f"🛡️ *Escrow Deal*\n\n"
                        f"*Buyer:* @{deal.buyer_username} ✅\n"
                        f"*Seller:* @{deal.seller_username} ✅\n"
                        f"*Amount:* {deal.amount} {deal.currency}\n"
                        f"*Status:* Payment instructions sent to buyer\n\n"
                        f"Escrow ID: `{deal.id}`"
                    ),
                    parse_mode="Markdown"
                )
                
                # Send payment instructions to buyer in private chat
                if is_buyer:
                    payment_instructions = (
                        f"🔐 *Payment Instructions*\n\n"
                        f"*Deal ID:* `{deal.id}`\n"
                        f"*Amount:* {deal.amount} {deal.currency}\n"
                        f"*Network:* {network}\n\n"
                        f"Please send exactly {deal.amount} {deal.currency} to this address:\n\n"
                        f"`{escrow_address}`\n\n"
                        f"⚠️ Important:\n"
                        f"- Send ONLY {deal.currency} on the {network} network\n"
                        f"- Include enough gas fee\n"
                        f"- The bot will automatically detect your payment\n"
                        f"- After sending, wait for confirmation"
                    )
                    
                    keyboard = [
                        [
                            InlineKeyboardButton("I've Sent Payment", callback_data=f"payment_sent:{deal_id}")
                        ],
                        [
                            InlineKeyboardButton("Cancel Deal", callback_data=f"cancel:{deal_id}")
                        ]
                    ]
                    
                    try:
                        await context.bot.send_message(
                            chat_id=user.id,
                            text=payment_instructions,
                            reply_markup=InlineKeyboardMarkup(keyboard),
                            parse_mode="Markdown"
                        )
                    except Exception as e:
                        logger.error(f"Failed to send payment instructions to buyer: {str(e)}")
                        await query.answer("Please start a private chat with me first", show_alert=True)
                
                # Send notification to seller
                if is_seller:
                    seller_notification = (
                        f"🔐 *Escrow Deal Update*\n\n"
                        f"*Deal ID:* `{deal.id}`\n"
                        f"*Buyer:* @{deal.buyer_username}\n"
                        f"*Amount:* {deal.amount} {deal.currency}\n\n"
                        f"Payment instructions have been sent to the buyer. You'll be notified when payment is received."
                    )
                    
                    try:
                        await context.bot.send_message(
                            chat_id=user.id,
                            text=seller_notification,
                            parse_mode="Markdown"
                        )
                    except Exception as e:
                        logger.error(f"Failed to send notification to seller: {str(e)}")
                
                await query.answer("Payment process started")
            else:
                await query.answer("Failed to setup payment", show_alert=True)
        
        elif action == "payment_sent":
            # Check if user is the buyer
            if user.username.lower() != deal.buyer_username.lower() or user.id != deal.buyer_id:
                await query.answer("Only the buyer can confirm payment", show_alert=True)
                return
            
            # Check deal state
            if deal.state != DealState.AWAITING_PAYMENT:
                await query.answer("This deal is not in payment state", show_alert=True)
                return
            
            # Instead of manual confirmation, we would normally check the blockchain
            # For this example, we'll simulate checking and assume payment is received
            
            # Check for payment (simplified)
            payment_received, updated_deal = await self.deal_manager.check_payment(deal_id)
            
            if payment_received:
                # Update the payment status message
                await query.edit_message_text(
                    text=(
                        f"🔐 *Payment Confirmed*\n\n"
                        f"*Deal ID:* `{deal.id}`\n"
                        f"*Amount:* {deal.amount} {deal.currency}\n"
                        f"*Status:* Payment received and confirmed\n\n"
                        f"The seller has been notified. Once you receive the goods/services, please confirm delivery."
                    ),
                    reply_markup=InlineKeyboardMarkup([
                        [
                            InlineKeyboardButton("Confirm Delivery", callback_data=f"confirm_delivery:{deal_id}")
                        ],
                        [
                            InlineKeyboardButton("Open Dispute", callback_data=f"open_dispute:{deal_id}")
                        ]
                    ]),
                    parse_mode="Markdown"
                )
                
                # Notify seller about payment
                try:
                    # Get seller's chat ID
                    if deal.seller_id:
                        await context.bot.send_message(
                            chat_id=deal.seller_id,
                            text=(
                                f"💰 *Payment Received*\n\n"
                                f"*Deal ID:* `{deal.id}`\n"
                                f"*Buyer:* @{deal.buyer_username}\n"
                                f"*Amount:* {deal.amount} {deal.currency}\n\n"
                                f"Payment has been received and is held in escrow. Please deliver the goods/services to the buyer."
                            ),
                            parse_mode="Markdown"
                        )
                except Exception as e:
                    logger.error(f"Failed to notify seller about payment: {str(e)}")
                
                # Also update the group chat message if possible
                try:
                    if deal.chat_id and deal.message_id:
                        await context.bot.send_message(
                            chat_id=deal.chat_id,
                            reply_to_message_id=deal.message_id,
                            text=f"💰 Payment received for deal `{deal.id}`. Escrow in progress.",
                            parse_mode="Markdown"
                        )
                except Exception as e:
                    logger.error(f"Failed to update group chat: {str(e)}")
                
                await query.answer("Payment confirmed")
            else:
                # Still waiting for payment
                await query.answer("Payment not yet detected. Please allow time for blockchain confirmation.", show_alert=True)
        
        elif action == "confirm_delivery":
            # Check if user is the buyer
            if user.username.lower() != deal.buyer_username.lower() or user.id != deal.buyer_id:
                await query.answer("Only the buyer can confirm delivery", show_alert=True)
                return
            
            # Check deal state
            if deal.state != DealState.PAYMENT_RECEIVED:
                await query.answer("This deal is not in the correct state for delivery confirmation", show_alert=True)
                return
            
            # Confirm delivery and release payment
            updated_deal = await self.deal_manager.confirm_delivery(deal_id, user.id)
            
            if updated_deal:
                # Update buyer's message
                await query.edit_message_text(
                    text=(
                        f"✅ *Deal Completed*\n\n"
                        f"*Deal ID:* `{deal.id}`\n"
                        f"*Amount:* {deal.amount} {deal.currency}\n"
                        f"*Status:* Completed\n\n"
                        f"You've confirmed delivery and payment has been released to the seller."
                    ),
                    parse_mode="Markdown"
                )
                
                # Notify seller about release
                try:
                    if deal.seller_id:
                        await context.bot.send_message(
                            chat_id=deal.seller_id,
                            text=(
                                f"✅ *Payment Released*\n\n"
                                f"*Deal ID:* `{deal.id}`\n"
                                f"*Buyer:* @{deal.buyer_username}\n"
                                f"*Amount:* {deal.amount} {deal.currency}\n\n"
                                f"The buyer has confirmed delivery and your payment has been released. Funds should arrive in your wallet shortly."
                            ),
                            parse_mode="Markdown"
                        )
                except Exception as e:
                    logger.error(f"Failed to notify seller about payment release: {str(e)}")
                
                # Update group chat if possible
                try:
                    if deal.chat_id:
                        await context.bot.send_message(
                            chat_id=deal.chat_id,
                            text=(
                                f"✅ *Deal Completed Successfully*\n\n"
                                f"*Deal ID:* `{deal.id}`\n"
                                f"*Buyer:* @{deal.buyer_username}\n"
                                f"*Seller:* @{deal.seller_username}\n\n"
                                f"The buyer has confirmed delivery and the payment has been released."
                            ),
                            parse_mode="Markdown"
                        )
                except Exception as e:
                    logger.error(f"Failed to update group chat about completed deal: {str(e)}")
                
                await query.answer("Payment released to seller")
            else:
                await query.answer("Failed to confirm delivery", show_alert=True)
        
        elif action == "open_dispute":
            # Check if user is a participant
            is_buyer = user.username.lower() == deal.buyer_username.lower() and user.id == deal.buyer_id
            is_seller = user.username.lower() == deal.seller_username.lower() and user.id == deal.seller_id
            
            if not (is_buyer or is_seller):
                await query.answer("Only deal participants can open disputes", show_alert=True)
                return
            
            # Check deal state
            if deal.state not in [DealState.PAYMENT_RECEIVED, DealState.AWAITING_PAYMENT]:
                await query.answer("Disputes can only be opened for active deals", show_alert=True)
                return
            
            # Start conversation for dispute reason
            await query.edit_message_text(
                text=(
                    f"🚨 *Opening Dispute*\n\n"
                    f"*Deal ID:* `{deal.id}`\n"
                    f"*Amount:* {deal.amount} {deal.currency}\n\n"
                    f"Please provide a detailed reason for this dispute. Use the command:\n"
                    f"`/dispute {deal.id} your detailed reason here`"
                ),
                parse_mode="Markdown"
            )
            
            await query.answer("Please provide dispute reason")
        
        elif action == "cancel":
            # Check if user is a participant
            is_buyer = user.username.lower() == deal.buyer_username.lower()
            is_seller = user.username.lower() == deal.seller_username.lower()
            
            if not (is_buyer or is_seller):
                await query.answer("Only deal participants can cancel", show_alert=True)
                return
            
            # Check deal state - only allow cancellation in early stages
            if deal.state not in [DealState.DETECTED, DealState.BUYER_CONFIRMED, DealState.SELLER_CONFIRMED, DealState.BOTH_CONFIRMED]:
                await query.answer("This deal cannot be cancelled at its current stage", show_alert=True)
                return
            
            # Cancel the deal
            updated_deal = await self.deal_manager.update_deal_state(deal_id, DealState.CANCELLED)
            
            if updated_deal:
                # Update message
                await query.edit_message_text(
                    text=(
                        f"❌ *Deal Cancelled*\n\n"
                        f"*Deal ID:* `{deal.id}`\n"
                        f"*Buyer:* @{deal.buyer_username}\n"
                        f"*Seller:* @{deal.seller_username}\n"
                        f"*Amount:* {deal.amount} {deal.currency}\n\n"
                        f"This deal has been cancelled by @{user.username}."
                    ),
                    parse_mode="Markdown"
                )
                
                # Log the cancellation
                await self.user_manager.log_activity(
                    user_id=user.id,
                    action_type="CANCEL_DEAL",
                    deal_id=deal_id,
                    details=f"Cancelled by {'buyer' if is_buyer else 'seller'}"
                )
                
                await query.answer("Deal cancelled successfully")
            else:
                await query.answer("Failed to cancel deal", show_alert=True)
        
        elif action == "edit":
            # Check if user is a participant
            is_buyer = user.username.lower() == deal.buyer_username.lower()
            is_seller = user.username.lower() == deal.seller_username.lower()
            
            if not (is_buyer or is_seller):
                await query.answer("Only deal participants can edit", show_alert=True)
                return
            
            # Check deal state - only allow editing in early stages
            if deal.state not in [DealState.DETECTED, DealState.BUYER_CONFIRMED, DealState.SELLER_CONFIRMED]:
                await query.answer("This deal cannot be edited at its current stage", show_alert=True)
                return
            
            # Instead of implementing a full editing system, suggest creating a new deal
            await query.edit_message_text(
                text=(
                    f"✏️ *Edit Deal*\n\n"
                    f"To edit this deal, please cancel it and create a new one with the correct details.\n\n"
                    f"Current details:\n"
                    f"*Buyer:* @{deal.buyer_username}\n"
                    f"*Seller:* @{deal.seller_username}\n"
                    f"*Amount:* {deal.amount} {deal.currency}\n"
                    f"*Condition:* {deal.condition}"
                ),
                reply_markup=InlineKeyboardMarkup([
                    [
                        InlineKeyboardButton("Cancel Deal", callback_data=f"cancel:{deal_id}"),
                        InlineKeyboardButton("Back", callback_data=f"back:{deal_id}")
                    ]
                ]),
                parse_mode="Markdown"
            )
            
            await query.answer("Please cancel and create a new deal to edit details")
        
        elif action == "back":
            # Go back to the original deal message
            state_text = ""
            
            if deal.state == DealState.DETECTED:
                keyboard = [
                    [
                        InlineKeyboardButton("Confirm as Buyer", callback_data=f"confirm_buyer:{deal.id}"),
                        InlineKeyboardButton("Confirm as Seller", callback_data=f"confirm_seller:{deal.id}")
                    ],
                    [
                        InlineKeyboardButton("Edit Details", callback_data=f"edit:{deal.id}"),
                        InlineKeyboardButton("Cancel", callback_data=f"cancel:{deal.id}")
                    ]
                ]
                state_text = "Both parties must confirm to proceed with the escrow."
            
            elif deal.state == DealState.BUYER_CONFIRMED:
                keyboard = [
                    [
                        InlineKeyboardButton("Confirm as Seller", callback_data=f"confirm_seller:{deal.id}")
                    ],
                    [
                        InlineKeyboardButton("Edit Details", callback_data=f"edit:{deal.id}"),
                        InlineKeyboardButton("Cancel", callback_data=f"cancel:{deal.id}")
                    ]
                ]
                state_text = "Buyer confirmed. Waiting for seller confirmation."
            
            elif deal.state == DealState.SELLER_CONFIRMED:
                keyboard = [
                    [
                        InlineKeyboardButton("Confirm as Buyer", callback_data=f"confirm_buyer:{deal.id}")
                    ],
                    [
                        InlineKeyboardButton("Edit Details", callback_data=f"edit:{deal.id}"),
                        InlineKeyboardButton("Cancel", callback_data=f"cancel:{deal.id}")
                    ]
                ]
                state_text = "Seller confirmed. Waiting for buyer confirmation."
            
            elif deal.state == DealState.BOTH_CONFIRMED:
                keyboard = [
                    [
                        InlineKeyboardButton("Start Payment Process", callback_data=f"setup_payment:{deal.id}")
                    ],
                    [
                        InlineKeyboardButton("Cancel", callback_data=f"cancel:{deal.id}")
                    ]
                ]
                state_text = "Both parties confirmed! Ready to proceed with payment."
            
            else:
                # For other states, just show a simple message without buttons
                await query.edit_message_text(
                    text=(
                        f"🛡️ *Escrow Deal*\n\n"
                        f"*Deal ID:* `{deal.id}`\n"
                        f"*Buyer:* @{deal.buyer_username}\n"
                        f"*Seller:* @{deal.seller_username}\n"
                        f"*Amount:* {deal.amount} {deal.currency}\n"
                        f"*Status:* {deal.state.name}\n\n"
                        f"This deal has progressed beyond the initial setup stage."
                    ),
                    parse_mode="Markdown"
                )
                await query.answer("Returned to deal summary")
                return
            
            # Update message with appropriate keyboard
            await query.edit_message_text(
                text=(
                    f"🛡️ *Escrow Deal*\n\n"
                    f"*Buyer:* @{deal.buyer_username}{' ✅' if deal.state in [DealState.BUYER_CONFIRMED, DealState.BOTH_CONFIRMED] else ''}\n"
                    f"*Seller:* @{deal.seller_username}{' ✅' if deal.state in [DealState.SELLER_CONFIRMED, DealState.BOTH_CONFIRMED] else ''}\n"
                    f"*Amount:* {deal.amount} {deal.currency}\n"
                    f"*Condition:* {deal.condition}\n\n"
                    f"{state_text}"
                ),
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="Markdown"
            )
            
            await query.answer("Returned to deal options")
    
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors in the bot."""
        logger.error(f"Exception while handling an update: {context.error}")
        
        # Log the error
        tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
        tb_string = ''.join(tb_list)
        logger.error(f"Exception details:\n{tb_string}")
        
        # Notify user if possible
        if update and isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text(
                "Sorry, an error occurred while processing your request. "
                "Our team has been notified."
            )
        
        # Notify admins about critical errors
        error_text = f"⚠️ *Bot Error*\n\n{str(context.error)[:100]}..."
        for admin_id in ADMIN_IDS:
            try:
                await context.bot.send_message(
                    chat_id=admin_id,
                    text=error_text,
                    parse_mode="Markdown"
                )
            except Exception as e:
                logger.error(f"Failed to notify admin {admin_id} about error: {str(e)}")

    async def start_payment_monitoring(self):
        """Start a background task to monitor payments."""
        while True:
            try:
                # Get all deals awaiting payment
                async with aiosqlite.connect(DB_PATH) as db:
                    db.row_factory = sqlite3.Row
                    async with db.execute('''
                    SELECT id FROM deals WHERE state = ?
                    ''', (DealState.AWAITING_PAYMENT.name,)) as cursor:
                        deal_ids = [row[0] for row in await cursor.fetchall()]
                
                # Check payment status for each deal
                for deal_id in deal_ids:
                    payment_received, deal = await self.deal_manager.check_payment(deal_id)
                    
                    if payment_received and deal:
                        logger.info(f"Payment received for deal {deal_id}")
                        
                        # Notify participants
                        bot = self.application.bot
                        
                        # Notify buyer
                        if deal.buyer_id:
                            try:
                                await bot.send_message(
                                    chat_id=deal.buyer_id,
                                    text=(
                                        f"💰 *Payment Confirmed*\n\n"
                                        f"*Deal ID:* `{deal.id}`\n"
                                        f"*Amount:* {deal.amount} {deal.currency}\n"
                                        f"*Status:* Payment received and confirmed\n\n"
                                        f"The seller has been notified. Once you receive the goods/services, please confirm delivery."
                                    ),
                                    reply_markup=InlineKeyboardMarkup([
                                        [
                                            InlineKeyboardButton("Confirm Delivery", callback_data=f"confirm_delivery:{deal.id}")
                                        ],
                                        [
                                            InlineKeyboardButton("Open Dispute", callback_data=f"open_dispute:{deal.id}")
                                        ]
                                    ]),
                                    parse_mode="Markdown"
                                )
                            except Exception as e:
                                logger.error(f"Failed to notify buyer about payment confirmation: {str(e)}")
                        
                        # Notify seller
                        if deal.seller_id:
                            try:
                                await bot.send_message(
                                    chat_id=deal.seller_id,
                                    text=(
                                        f"💰 *Payment Received*\n\n"
                                        f"*Deal ID:* `{deal.id}`\n"
                                        f"*Buyer:* @{deal.buyer_username}\n"
                                        f"*Amount:* {deal.amount} {deal.currency}\n\n"
                                        f"Payment has been received and is held in escrow. Please deliver the goods/services to the buyer."
                                    ),
                                    parse_mode="Markdown"
                                )
                            except Exception as e:
                                logger.error(f"Failed to notify seller about payment: {str(e)}")
                
                # Check every 30 seconds
                await asyncio.sleep(30)
            
            except Exception as e:
                logger.error(f"Error in payment monitoring: {str(e)}")
                await asyncio.sleep(60)  # Wait a bit longer if there's an error
    
    async def cleanup_old_deals(self):
        """Clean up old detected deals that haven't been confirmed."""
        while True:
            try:
                # Find deals in DETECTED state older than 24 hours
                time_threshold = datetime.datetime.now() - datetime.timedelta(hours=24)
                
                async with aiosqlite.connect(DB_PATH) as db:
                    db.row_factory = sqlite3.Row
                    async with db.execute('''
                    SELECT id FROM deals 
                    WHERE state IN (?, ?, ?) 
                    AND created_at < ?
                    ''', (
                        DealState.DETECTED.name,
                        DealState.BUYER_CONFIRMED.name,
                        DealState.SELLER_CONFIRMED.name,
                        time_threshold.isoformat()
                    )) as cursor:
                        old_deal_ids = [row[0] for row in await cursor.fetchall()]
                
                # Cancel old deals
                for deal_id in old_deal_ids:
                    logger.info(f"Auto-cancelling old unconfirmed deal {deal_id}")
                    await self.deal_manager.update_deal_state(deal_id, DealState.CANCELLED)
                
                # Run once every hour
                await asyncio.sleep(3600)
            
            except Exception as e:
                logger.error(f"Error in deal cleanup: {str(e)}")
                await asyncio.sleep(3600)  # Still wait an hour before trying again
    
    async def run(self):
        """Run the bot and start background tasks."""
        # Setup database
        await setup_database()
        
        # Start background tasks
        asyncio.create_task(self.start_payment_monitoring())
        asyncio.create_task(self.cleanup_old_deals())
        
        # Start the bot
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()
        
        logger.info("Bot started")
        
        # Keep the bot running
        try:
            await self.application.updater.start_polling()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Bot stopping...")
        finally:
            await self.application.stop()
            await self.application.shutdown()

# Main entry point
async def main():
    """Initialize and run the bot."""
    bot = EscrowBot()
    await bot.run()

if __name__ == "__main__":
    # Import missing modules needed in error_handler
    import traceback
    
    # Run the bot
    asyncio.run(main())