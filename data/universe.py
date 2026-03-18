"""
data/universe.py — Complete Indian market universe definition

INTENT:
    Defines every tradeable instrument we backtest across: equities (all NSE/BSE
    listed), indices (broad + sectoral), commodities (MCX), and F&O (futures &
    options). This is the authoritative list — the backtester, scanner, and agent
    all pull from here.

IMPACT:
    Adding a new instrument here makes it available to every downstream system
    automatically. Removing one stops it from being traded without touching
    any other file.

FUNCTIONS:
    - get_equity_universe(): All NSE-listed equities
    - get_index_universe(): Broad + sectoral indices
    - get_commodity_universe(): MCX commodities
    - get_fno_universe(): F&O eligible stocks
    - get_full_universe(): Everything combined

OWNED BY: Phase 1 — Data Pipeline
LAST UPDATED: 2026-03-18
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AssetClass(Enum):
    EQUITY = "equity"
    INDEX = "index"
    COMMODITY = "commodity"
    FUTURES = "futures"
    OPTIONS = "options"
    ETF = "etf"


class Exchange(Enum):
    NSE = "NSE"
    BSE = "BSE"
    MCX = "MCX"
    NFO = "NFO"   # NSE F&O
    BFO = "BFO"   # BSE F&O


@dataclass
class Instrument:
    symbol: str
    name: str
    asset_class: AssetClass
    exchange: Exchange
    dhan_security_id: Optional[str] = None
    yfinance_ticker: Optional[str] = None
    lot_size: int = 1
    tick_size: float = 0.05
    segment: Optional[str] = None   # IT, Banking, Auto, etc.


# ─────────────────────────────────────────────────────────────────
# NIFTY BROAD INDICES
# ─────────────────────────────────────────────────────────────────
BROAD_INDICES = [
    Instrument("NIFTY",       "Nifty 50",            AssetClass.INDEX, Exchange.NSE, yfinance_ticker="^NSEI",    segment="Broad"),
    Instrument("BANKNIFTY",   "Nifty Bank",          AssetClass.INDEX, Exchange.NSE, yfinance_ticker="^NSEBANK", segment="Banking"),
    Instrument("FINNIFTY",    "Nifty Financial Svcs",AssetClass.INDEX, Exchange.NSE, yfinance_ticker="NIFTY_FIN_SERVICE.NS", segment="Financials"),
    Instrument("MIDCPNIFTY",  "Nifty Midcap 50",     AssetClass.INDEX, Exchange.NSE, yfinance_ticker="NIFTY_MID_SELECT.NS", segment="Midcap"),
    Instrument("NIFTY100",    "Nifty 100",           AssetClass.INDEX, Exchange.NSE, yfinance_ticker="^CNX100",  segment="Broad"),
    Instrument("NIFTY200",    "Nifty 200",           AssetClass.INDEX, Exchange.NSE, yfinance_ticker="^CNX200",  segment="Broad"),
    Instrument("NIFTY500",    "Nifty 500",           AssetClass.INDEX, Exchange.NSE, yfinance_ticker="^CNX500",  segment="Broad"),
    Instrument("NIFTYMIDCAP", "Nifty Midcap 100",    AssetClass.INDEX, Exchange.NSE, yfinance_ticker="NIFTY_MIDCAP_100.NS", segment="Midcap"),
    Instrument("NIFTYSMALLCAP","Nifty Smallcap 100", AssetClass.INDEX, Exchange.NSE, yfinance_ticker="NIFTY_SMLCAP_100.NS", segment="Smallcap"),
    Instrument("INDIA_VIX",   "India VIX",           AssetClass.INDEX, Exchange.NSE, yfinance_ticker="^INDIAVIX", segment="Volatility"),
]

# ─────────────────────────────────────────────────────────────────
# SECTORAL INDICES
# ─────────────────────────────────────────────────────────────────
SECTORAL_INDICES = [
    Instrument("NIFTYIT",     "Nifty IT",            AssetClass.INDEX, Exchange.NSE, yfinance_ticker="^CNXIT",   segment="IT"),
    Instrument("NIFTYPHARMA", "Nifty Pharma",        AssetClass.INDEX, Exchange.NSE, yfinance_ticker="NIFTY_PHARMA.NS", segment="Pharma"),
    Instrument("NIFTYAUTO",   "Nifty Auto",          AssetClass.INDEX, Exchange.NSE, yfinance_ticker="NIFTY_AUTO.NS", segment="Auto"),
    Instrument("NIFTYFMCG",   "Nifty FMCG",          AssetClass.INDEX, Exchange.NSE, yfinance_ticker="NIFTY_FMCG.NS", segment="FMCG"),
    Instrument("NIFTYMETAL",  "Nifty Metal",         AssetClass.INDEX, Exchange.NSE, yfinance_ticker="NIFTY_METAL.NS", segment="Metal"),
    Instrument("NIFTYENERGY", "Nifty Energy",        AssetClass.INDEX, Exchange.NSE, yfinance_ticker="NIFTY_ENERGY.NS", segment="Energy"),
    Instrument("NIFTYREALTY", "Nifty Realty",        AssetClass.INDEX, Exchange.NSE, yfinance_ticker="NIFTY_REALTY.NS", segment="Realty"),
    Instrument("NIFTYINFRA",  "Nifty Infra",         AssetClass.INDEX, Exchange.NSE, yfinance_ticker="NIFTY_INFRA.NS", segment="Infra"),
    Instrument("NIFTYPSUBANK","Nifty PSU Bank",      AssetClass.INDEX, Exchange.NSE, yfinance_ticker="NIFTY_PSU_BANK.NS", segment="PSU Banking"),
    Instrument("NIFTYMEDIUM", "Nifty Midcap Select", AssetClass.INDEX, Exchange.NSE, yfinance_ticker="NIFTY_MID_SELECT.NS", segment="Midcap"),
    Instrument("NIFTYDEFENCE","Nifty India Defence", AssetClass.INDEX, Exchange.NSE, yfinance_ticker="NIFTY_INDIA_DEFENCE.NS", segment="Defence"),
    Instrument("NIFTYHEALTHCARE","Nifty Healthcare",  AssetClass.INDEX, Exchange.NSE, yfinance_ticker="NIFTY_HEALTHCARE.NS", segment="Healthcare"),
]

# ─────────────────────────────────────────────────────────────────
# NIFTY 50 CONSTITUENTS (F&O eligible, most liquid)
# ─────────────────────────────────────────────────────────────────
NIFTY50_STOCKS = [
    Instrument("RELIANCE",  "Reliance Industries",  AssetClass.EQUITY, Exchange.NSE, dhan_security_id="2885",  yfinance_ticker="RELIANCE.NS",  lot_size=250,  segment="Energy"),
    Instrument("TCS",       "Tata Consultancy",     AssetClass.EQUITY, Exchange.NSE, dhan_security_id="11536", yfinance_ticker="TCS.NS",       lot_size=150,  segment="IT"),
    Instrument("HDFCBANK",  "HDFC Bank",            AssetClass.EQUITY, Exchange.NSE, dhan_security_id="1333",  yfinance_ticker="HDFCBANK.NS",  lot_size=550,  segment="Banking"),
    Instrument("INFY",      "Infosys",              AssetClass.EQUITY, Exchange.NSE, dhan_security_id="1594",  yfinance_ticker="INFY.NS",      lot_size=400,  segment="IT"),
    Instrument("ICICIBANK", "ICICI Bank",           AssetClass.EQUITY, Exchange.NSE, dhan_security_id="4963",  yfinance_ticker="ICICIBANK.NS", lot_size=700,  segment="Banking"),
    Instrument("HINDUNILVR","Hindustan Unilever",   AssetClass.EQUITY, Exchange.NSE, dhan_security_id="1394",  yfinance_ticker="HINDUNILVR.NS",lot_size=300,  segment="FMCG"),
    Instrument("ITC",       "ITC Ltd",              AssetClass.EQUITY, Exchange.NSE, dhan_security_id="1660",  yfinance_ticker="ITC.NS",       lot_size=1600, segment="FMCG"),
    Instrument("SBIN",      "State Bank of India",  AssetClass.EQUITY, Exchange.NSE, dhan_security_id="3045",  yfinance_ticker="SBIN.NS",      lot_size=1500, segment="Banking"),
    Instrument("BHARTIARTL","Bharti Airtel",        AssetClass.EQUITY, Exchange.NSE, dhan_security_id="10604", yfinance_ticker="BHARTIARTL.NS",lot_size=500,  segment="Telecom"),
    Instrument("KOTAKBANK", "Kotak Mahindra Bank",  AssetClass.EQUITY, Exchange.NSE, dhan_security_id="1922",  yfinance_ticker="KOTAKBANK.NS", lot_size=400,  segment="Banking"),
    Instrument("LT",        "Larsen & Toubro",      AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="LT.NS",        lot_size=150,  segment="Infra"),
    Instrument("HCLTECH",   "HCL Technologies",     AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="HCLTECH.NS",   lot_size=350,  segment="IT"),
    Instrument("AXISBANK",  "Axis Bank",            AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="AXISBANK.NS",  lot_size=625,  segment="Banking"),
    Instrument("WIPRO",     "Wipro",                AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="WIPRO.NS",     lot_size=1500, segment="IT"),
    Instrument("MARUTI",    "Maruti Suzuki",        AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="MARUTI.NS",    lot_size=100,  segment="Auto"),
    Instrument("SUNPHARMA", "Sun Pharmaceutical",   AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="SUNPHARMA.NS", lot_size=350,  segment="Pharma"),
    Instrument("TATAMOTORS","Tata Motors",          AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="TATAMOTORS.NS",lot_size=1400, segment="Auto"),
    Instrument("TATASTEEL", "Tata Steel",           AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="TATASTEEL.NS", lot_size=5500, segment="Metal"),
    Instrument("NESTLEIND", "Nestle India",         AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="NESTLEIND.NS", lot_size=40,   segment="FMCG"),
    Instrument("POWERGRID", "Power Grid",           AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="POWERGRID.NS", lot_size=2700, segment="Energy"),
    Instrument("NTPC",      "NTPC Ltd",             AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="NTPC.NS",      lot_size=2250, segment="Energy"),
    Instrument("ONGC",      "ONGC",                 AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="ONGC.NS",      lot_size=1925, segment="Energy"),
    Instrument("COALINDIA", "Coal India",           AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="COALINDIA.NS", lot_size=1400, segment="Metal"),
    Instrument("JSWSTEEL",  "JSW Steel",            AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="JSWSTEEL.NS",  lot_size=600,  segment="Metal"),
    Instrument("BAJFINANCE","Bajaj Finance",        AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="BAJFINANCE.NS",lot_size=125,  segment="Financials"),
    Instrument("BAJAJFINSV","Bajaj Finserv",        AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="BAJAJFINSV.NS",lot_size=500,  segment="Financials"),
    Instrument("HDFCLIFE",  "HDFC Life Insurance",  AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="HDFCLIFE.NS",  lot_size=1100, segment="Financials"),
    Instrument("SBILIFE",   "SBI Life Insurance",   AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="SBILIFE.NS",   lot_size=750,  segment="Financials"),
    Instrument("TECHM",     "Tech Mahindra",        AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="TECHM.NS",     lot_size=600,  segment="IT"),
    Instrument("ULTRACEMCO","UltraTech Cement",     AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="ULTRACEMCO.NS",lot_size=100,  segment="Infra"),
    Instrument("TITAN",     "Titan Company",        AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="TITAN.NS",     lot_size=375,  segment="Consumer"),
    Instrument("ASIANPAINT","Asian Paints",         AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="ASIANPAINT.NS",lot_size=300,  segment="Consumer"),
    Instrument("DRREDDY",   "Dr Reddy's Labs",      AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="DRREDDY.NS",   lot_size=125,  segment="Pharma"),
    Instrument("CIPLA",     "Cipla",                AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="CIPLA.NS",     lot_size=650,  segment="Pharma"),
    Instrument("DIVISLAB",  "Divi's Laboratories",  AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="DIVISLAB.NS",  lot_size=200,  segment="Pharma"),
    Instrument("APOLLOHOSP","Apollo Hospitals",     AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="APOLLOHOSP.NS",lot_size=125,  segment="Healthcare"),
    Instrument("ADANIPORTS","Adani Ports",          AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="ADANIPORTS.NS",lot_size=600,  segment="Infra"),
    Instrument("ADANIENT",  "Adani Enterprises",    AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="ADANIENT.NS",  lot_size=250,  segment="Conglomerate"),
    Instrument("GRASIM",    "Grasim Industries",    AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="GRASIM.NS",    lot_size=475,  segment="Diversified"),
    Instrument("INDUSINDBK","IndusInd Bank",        AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="INDUSINDBK.NS",lot_size=500,  segment="Banking"),
    Instrument("BPCL",      "BPCL",                 AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="BPCL.NS",      lot_size=1800, segment="Energy"),
    Instrument("EICHERMOT", "Eicher Motors",        AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="EICHERMOT.NS", lot_size=175,  segment="Auto"),
    Instrument("HEROMOTOCO","Hero MotoCorp",        AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="HEROMOTOCO.NS",lot_size=300,  segment="Auto"),
    Instrument("TATACONSUM","Tata Consumer Products",AssetClass.EQUITY,Exchange.NSE, yfinance_ticker="TATACONSUM.NS",lot_size=875,  segment="FMCG"),
    Instrument("BRITANNIA", "Britannia Industries", AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="BRITANNIA.NS", lot_size=200,  segment="FMCG"),
    Instrument("M&M",       "Mahindra & Mahindra",  AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="M&M.NS",       lot_size=700,  segment="Auto"),
    Instrument("SHREECEM",  "Shree Cement",         AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="SHREECEM.NS",  lot_size=25,   segment="Infra"),
    Instrument("BAJAJ-AUTO","Bajaj Auto",           AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="BAJAJ-AUTO.NS",lot_size=250,  segment="Auto"),
    Instrument("HINDALCO",  "Hindalco Industries",  AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="HINDALCO.NS",  lot_size=1400, segment="Metal"),
    Instrument("VEDL",      "Vedanta",              AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="VEDL.NS",      lot_size=2500, segment="Metal"),
]

# ─────────────────────────────────────────────────────────────────
# NIFTY NEXT 50 (high-growth mid-large caps)
# ─────────────────────────────────────────────────────────────────
NIFTY_NEXT50_STOCKS = [
    Instrument("ZOMATO",    "Zomato",               AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="ZOMATO.NS",   lot_size=3750, segment="Consumer Tech"),
    Instrument("PAYTM",     "One97 Communications", AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="PAYTM.NS",    lot_size=2000, segment="Fintech"),
    Instrument("NYKAA",     "FSN E-Commerce (Nykaa)",AssetClass.EQUITY,Exchange.NSE, yfinance_ticker="NYKAA.NS",    lot_size=4600, segment="Consumer Tech"),
    Instrument("DMART",     "Avenue Supermarts",    AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="DMART.NS",    lot_size=75,   segment="Retail"),
    Instrument("IRCTC",     "IRCTC",                AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="IRCTC.NS",    lot_size=875,  segment="Travel"),
    Instrument("HAL",       "Hindustan Aeronautics",AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="HAL.NS",      lot_size=150,  segment="Defence"),
    Instrument("BEL",       "Bharat Electronics",   AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="BEL.NS",      lot_size=4800, segment="Defence"),
    Instrument("ICICIGI",   "ICICI Lombard",        AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="ICICIGI.NS",  lot_size=400,  segment="Insurance"),
    Instrument("PIDILITIND","Pidilite Industries",  AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="PIDILITIND.NS",lot_size=250, segment="Chemicals"),
    Instrument("SIEMENS",   "Siemens India",        AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="SIEMENS.NS",  lot_size=125,  segment="Capital Goods"),
    Instrument("ABB",       "ABB India",            AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="ABB.NS",      lot_size=150,  segment="Capital Goods"),
    Instrument("TRENT",     "Trent",                AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="TRENT.NS",    lot_size=350,  segment="Retail"),
    Instrument("MUTHOOTFIN","Muthoot Finance",      AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="MUTHOOTFIN.NS",lot_size=450, segment="Financials"),
    Instrument("POLYCAB",   "Polycab India",        AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="POLYCAB.NS",  lot_size=250,  segment="Capital Goods"),
    Instrument("GODREJCP",  "Godrej Consumer",      AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="GODREJCP.NS", lot_size=500,  segment="FMCG"),
    Instrument("MARICO",    "Marico",               AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="MARICO.NS",   lot_size=1200, segment="FMCG"),
    Instrument("COLPAL",    "Colgate Palmolive",    AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="COLPAL.NS",   lot_size=700,  segment="FMCG"),
    Instrument("DABUR",     "Dabur India",          AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="DABUR.NS",    lot_size=1250, segment="FMCG"),
    Instrument("MCDOWELL-N","United Spirits",       AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="MCDOWELL-N.NS",lot_size=500, segment="Consumer"),
    Instrument("INDIGO",    "IndiGo (InterGlobe)",  AssetClass.EQUITY, Exchange.NSE, yfinance_ticker="INDIGO.NS",   lot_size=300,  segment="Aviation"),
]

# ─────────────────────────────────────────────────────────────────
# MCX COMMODITIES
# ─────────────────────────────────────────────────────────────────
COMMODITIES = [
    Instrument("GOLD",      "Gold",                 AssetClass.COMMODITY, Exchange.MCX, yfinance_ticker="GC=F",  lot_size=100,  tick_size=1.0,  segment="Precious Metal"),
    Instrument("SILVER",    "Silver",               AssetClass.COMMODITY, Exchange.MCX, yfinance_ticker="SI=F",  lot_size=30000,tick_size=1.0,  segment="Precious Metal"),
    Instrument("CRUDEOIL",  "Crude Oil",            AssetClass.COMMODITY, Exchange.MCX, yfinance_ticker="CL=F",  lot_size=100,  tick_size=1.0,  segment="Energy"),
    Instrument("NATURALGAS","Natural Gas",          AssetClass.COMMODITY, Exchange.MCX, yfinance_ticker="NG=F",  lot_size=1250, tick_size=0.1,  segment="Energy"),
    Instrument("COPPER",    "Copper",               AssetClass.COMMODITY, Exchange.MCX, yfinance_ticker="HG=F",  lot_size=2500, tick_size=0.05, segment="Base Metal"),
    Instrument("ALUMINIUM", "Aluminium",            AssetClass.COMMODITY, Exchange.MCX, yfinance_ticker="ALI=F", lot_size=5000, tick_size=0.05, segment="Base Metal"),
    Instrument("ZINC",      "Zinc",                 AssetClass.COMMODITY, Exchange.MCX, yfinance_ticker="ZNC=F", lot_size=5000, tick_size=0.05, segment="Base Metal"),
    Instrument("LEAD",      "Lead",                 AssetClass.COMMODITY, Exchange.MCX, yfinance_ticker="LE=F",  lot_size=5000, tick_size=0.05, segment="Base Metal"),
    Instrument("NICKEL",    "Nickel",               AssetClass.COMMODITY, Exchange.MCX, yfinance_ticker="NI=F",  lot_size=1500, tick_size=0.1,  segment="Base Metal"),
    Instrument("COTTON",    "Cotton",               AssetClass.COMMODITY, Exchange.MCX, yfinance_ticker="CT=F",  lot_size=25,   tick_size=10.0, segment="Agri"),
]

# ─────────────────────────────────────────────────────────────────
# LIQUID ETFs (trade like equities, index exposure)
# ─────────────────────────────────────────────────────────────────
LIQUID_ETFS = [
    Instrument("NIFTYBEES",  "Nippon Nifty BeES ETF",   AssetClass.ETF, Exchange.NSE, yfinance_ticker="NIFTYBEES.NS",  segment="Broad"),
    Instrument("BANKBEES",   "Nippon Bank BeES ETF",    AssetClass.ETF, Exchange.NSE, yfinance_ticker="BANKBEES.NS",   segment="Banking"),
    Instrument("GOLDBEES",   "Nippon Gold BeES ETF",    AssetClass.ETF, Exchange.NSE, yfinance_ticker="GOLDBEES.NS",   segment="Gold"),
    Instrument("JUNIORBEES", "Nippon Junior BeES ETF",  AssetClass.ETF, Exchange.NSE, yfinance_ticker="JUNIORBEES.NS", segment="Midcap"),
    Instrument("ITBEES",     "Nippon IT BeES ETF",      AssetClass.ETF, Exchange.NSE, yfinance_ticker="ITBEES.NS",     segment="IT"),
    Instrument("CPSEETF",    "CPSE ETF",                AssetClass.ETF, Exchange.NSE, yfinance_ticker="CPSEETF.NS",    segment="PSU"),
]


def get_equity_universe() -> list[Instrument]:
    return NIFTY50_STOCKS + NIFTY_NEXT50_STOCKS


def get_index_universe() -> list[Instrument]:
    return BROAD_INDICES + SECTORAL_INDICES


def get_commodity_universe() -> list[Instrument]:
    return COMMODITIES


def get_fno_universe() -> list[Instrument]:
    """F&O eligible: Nifty 50 stocks + major indices."""
    return NIFTY50_STOCKS + [
        i for i in BROAD_INDICES if i.symbol in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY")
    ]


def get_etf_universe() -> list[Instrument]:
    return LIQUID_ETFS


def get_full_universe() -> list[Instrument]:
    return (
        get_equity_universe()
        + get_index_universe()
        + get_commodity_universe()
        + get_etf_universe()
    )


def get_universe_by_segment(segment: str) -> list[Instrument]:
    return [i for i in get_full_universe() if i.segment == segment]


ALL_SEGMENTS = sorted(set(
    i.segment for i in get_full_universe() if i.segment
))
