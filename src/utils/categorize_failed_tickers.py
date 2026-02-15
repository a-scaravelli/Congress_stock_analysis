"""
Categorize failed tickers by their fate: acquired, bankrupt, went private, etc.

Combines:
1. Professor's delisted_tickers_research.csv
2. Manually researched tickers (web searches from this project)
3. Pattern-based inference (ADR suffixes, data errors, etc.)

Output: data/failed_tickers_map.csv
"""

import csv
import os
import polars as pl

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "execution_log.csv")
SOURCE_FILE = os.path.join(PROJECT_ROOT, "data", "trades", "congress_trades_full.parquet")
PROFESSOR_FILE = os.path.join(PROJECT_ROOT, "data", "congress_trades_professor", "delisted_tickers_research.csv")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "failed_tickers_map.csv")

# --- Categories ---
# ACQUIRED_PUBLIC   = acquired by a publicly traded company (successor ticker exists)
# ACQUIRED_PRIVATE  = acquired by private equity or taken private
# BANKRUPT          = filed for bankruptcy / liquidated
# MERGED            = merged into another entity (public)
# WENT_PRIVATE      = taken private (LBO, management buyout, etc.)
# TICKER_CHANGED    = ticker symbol changed but company still trades
# SPAC_FAILED       = SPAC that failed or liquidated
# DATA_ERROR        = not a real ticker (typo, data quality issue)
# ADR_FOREIGN       = foreign ADR or international listing (often illiquid/delisted)
# YFINANCE_ISSUE    = company still trades but yfinance can't find it
# UNKNOWN           = insufficient information to categorize

# ============================================================
# MANUAL RESEARCH DATABASE
# Tickers researched via web searches in this project
# Format: ticker -> (category, date, acquirer_or_notes)
# ============================================================
MANUAL_RESEARCH = {
    # --- ACQUIRED BY PUBLIC COMPANY ---
    "CBS": ("MERGED", "2019-12-04", "Merged with Viacom to form ViacomCBS (now PARA)"),
    "FEYE": ("ACQUIRED_PUBLIC", "2022-10-11", "Acquired by Google/Mandiant (GOOG)"),
    "CATM": ("ACQUIRED_PUBLIC", "2020-06-01", "Cardtronics acquired by NCR Corporation (NCR)"),
    "XLNX": ("ACQUIRED_PUBLIC", "2022-02-14", "Xilinx acquired by AMD"),
    "CLGX": ("ACQUIRED_PUBLIC", "2021-06-16", "CoreLogic acquired by Stone Point/Insight Partners (private)"),
    "CDK": ("ACQUIRED_PRIVATE", "2022-07-06", "CDK Global acquired by Brookfield Business Partners"),
    "USG": ("ACQUIRED_PUBLIC", "2019-04-15", "USG acquired by Knauf (private German company)"),
    "STOR": ("ACQUIRED_PRIVATE", "2022-09-15", "STORE Capital acquired by GIC/Oak Street (private)"),
    "CTCT": ("ACQUIRED_PRIVATE", "2015-07-01", "Constant Contact acquired by Endurance International (private)"),
    "ERJ": ("YFINANCE_ISSUE", None, "Embraer still trades as ERJ on NYSE"),
    "WPZ": ("MERGED", "2018-08-13", "Williams Partners merged into Williams Companies (WMB)"),
    "ETE": ("MERGED", "2018-10-19", "Energy Transfer Equity merged into Energy Transfer (ET)"),
    "ENBL": ("MERGED", "2020-12-14", "Enable Midstream merged into Energy Transfer (ET)"),
    "PSXP": ("MERGED", "2021-03-19", "Phillips 66 Partners merged into Phillips 66 (PSX)"),
    "NBLX": ("MERGED", "2020-10-03", "Noble Midstream merged into Chevron (CVX)"),
    "EQM": ("MERGED", "2020-06-17", "EQT Midstream merged into Equitrans Midstream (ETRN)"),
    "CEQP": ("MERGED", "2022-11-01", "Crestwood Equity merged into Crestwood Midstream"),
    "POT": ("MERGED", "2018-01-01", "Potash Corp merged with Agrium to form Nutrien (NTR)"),
    "DNB": ("YFINANCE_ISSUE", None, "Dun & Bradstreet trades as DNB on NYSE - likely yfinance issue"),
    "LSXMK": ("MERGED", "2024-09-09", "Liberty SiriusXM restructured into Sirius XM (SIRI)"),
    "VSLR": ("MERGED", "2020-10-05", "Vivint Solar merged into Sunrun (RUN)"),
    "AVP": ("ACQUIRED_PUBLIC", "2020-01-03", "Avon Products acquired by Natura & Co (NTCO)"),
    "CLR": ("WENT_PRIVATE", "2022-11-22", "Continental Resources taken private by Harold Hamm family"),
    "HHC": ("YFINANCE_ISSUE", None, "Howard Hughes Corp trades as HHH after 2023 restructuring"),
    "WFT": ("BANKRUPT", "2019-07-01", "Weatherford International filed Chapter 11 bankruptcy"),
    "YNDX": ("WENT_PRIVATE", "2024-07-15", "Yandex delisted, restructured as Nebius Group (NBIS)"),
    "SGH": ("YFINANCE_ISSUE", None, "SMART Global Holdings still trades as SGH"),
    "VBTX": ("YFINANCE_ISSUE", None, "Veritex Holdings still trades as VBTX on NASDAQ"),
    "VSM": ("YFINANCE_ISSUE", None, "Versum Materials acquired by Merck KGaA (2019-10-07)"),
    "SNE": ("TICKER_CHANGED", "2021-04-01", "Sony changed ticker from SNE to SONY"),
    "WDR": ("ACQUIRED_PRIVATE", "2021-04-30", "Waddell & Reed acquired by Macquarie Asset Management"),
    "HYH": ("YFINANCE_ISSUE", None, "Halyard Health - acquired by Owens & Minor (OMI) 2018"),
    "DPSGY": ("MERGED", "2018-07-09", "Dr Pepper Snapple merged into Keurig Dr Pepper (KDP)"),
    "PKI": ("TICKER_CHANGED", "2023-03-13", "PerkinElmer renamed to Revvity, ticker changed to RVTY"),
    "JWN": ("WENT_PRIVATE", "2025-03-24", "Nordstrom taken private by Nordstrom family & El Puerto de Liverpool"),
    "BRK.B": ("YFINANCE_ISSUE", None, "Berkshire Hathaway trades as BRK-B on yfinance (handled by mapping)"),
    "WBA": ("WENT_PRIVATE", "2024-12-23", "Walgreens Boots Alliance taken private by Sycamore Partners"),
    "FI": ("YFINANCE_ISSUE", None, "Fiserv currently trades as FI on NYSE (formerly FISV)"),
    "BLL": ("YFINANCE_ISSUE", None, "Ball Corporation still trades as BLL on NYSE"),
    "CONN": ("BANKRUPT", "2024-07-23", "Conn's filed Chapter 11 bankruptcy"),
    "SKX": ("WENT_PRIVATE", "2025-07-15", "Skechers taken private by 3G Capital"),
    "TWTR": ("WENT_PRIVATE", "2022-10-27", "Twitter acquired by Elon Musk / X Corp"),
    "SIVB": ("BANKRUPT", "2023-03-10", "Silicon Valley Bank seized by FDIC, sold to First Citizens (FCNCA)"),
    "CTXS": ("ACQUIRED_PRIVATE", "2022-09-30", "Citrix acquired by Vista Equity Partners & Evergreen Coast Capital"),
    "NLSN": ("ACQUIRED_PRIVATE", "2022-10-11", "Nielsen acquired by Evergreen Coast Capital & Brookfield"),
    "QLIK": ("ACQUIRED_PRIVATE", "2016-08-01", "Qlik acquired by Thoma Bravo"),
    "RCII": ("ACQUIRED_PRIVATE", "2024-06-18", "Rent-A-Center acquired by CD&R and TowerBrook Capital"),
    "COUP": ("ACQUIRED_PRIVATE", "2023-02-13", "Coupa Software acquired by Thoma Bravo"),
    "MAXR": ("ACQUIRED_PRIVATE", "2023-05-16", "Maxar Technologies acquired by Advent International"),
    "NVEE": ("ACQUIRED_PRIVATE", "2025-04-01", "NV5 Global acquired by Acuren Corporation"),
    "ALTM": ("ACQUIRED_PRIVATE", "2023-12-01", "Altus Midstream merged/acquired"),
    "DWAC": ("MERGED", "2024-03-25", "Digital World Acquisition merged with Trump Media (DJT)"),
    "CBL": ("BANKRUPT", "2020-11-01", "CBL & Associates filed Chapter 11 bankruptcy"),
    "CHK": ("BANKRUPT", "2020-06-28", "Chesapeake Energy filed Chapter 11 bankruptcy, emerged 2021"),
    "OPI": ("BANKRUPT", "2024-06-07", "Office Properties Income Trust filed Chapter 11 bankruptcy"),
    "AKRX": ("BANKRUPT", "2020-05-20", "Akorn filed Chapter 11 bankruptcy"),
    "TWOU": ("BANKRUPT", "2024-07-31", "2U filed Chapter 11 bankruptcy"),
    "NKLA": ("BANKRUPT", "2024-11-19", "Nikola filed Chapter 11 bankruptcy"),
    "ABCO": ("ACQUIRED_PUBLIC", "2020-08-06", "Advisory Board Company acquired by Optum/UnitedHealth (UNH)"),
    "PSTH": ("SPAC_FAILED", "2022-09-01", "Pershing Square Tontine Holdings SPAC dissolved"),
    "PBFX": ("MERGED", "2024-10-01", "PBF Logistics merged into PBF Energy (PBF)"),
    "GLOP": ("ACQUIRED_PRIVATE", "2024-01-01", "GasLog Partners acquired by GasLog (private)"),
    "CUTR": ("ACQUIRED_PRIVATE", "2024-01-25", "Cutera went private / delisted"),
    "EQC": ("MERGED", "2023-10-01", "Equity Commonwealth liquidated/dissolved assets"),
    "DISH": ("MERGED", "2024-12-31", "DISH Network merged into EchoStar (SATS)"),
    "TACO": ("YFINANCE_ISSUE", None, "Del Taco went private via Jack in the Box (JACK) acquisition 2022"),
    "WW": ("YFINANCE_ISSUE", None, "WW International (Weight Watchers) still trades but may have been delisted"),
    "VSM": ("ACQUIRED_PUBLIC", "2019-10-07", "Versum Materials acquired by Merck KGaA (private)"),

    # --- Researched in prior sessions ---
    "DCPH": ("ACQUIRED_PRIVATE", "2024-06-05", "Deciphera acquired by ONO Pharmaceutical (private/Japan)"),
    "TLRD": ("BANKRUPT", "2020-08-02", "Tailored Brands filed Chapter 7 bankruptcy"),
    "LOGM": ("WENT_PRIVATE", "2020-08-31", "LogMeIn taken private by Francisco Partners & Evergreen Coast Capital"),
    "AEGN": ("WENT_PRIVATE", "2018-12-18", "Aegion Corporation taken private by New Mountain Capital"),

    # --- More researched tickers by trade count ---
    "AGU": ("MERGED", "2018-01-01", "Agrium merged with PotashCorp to form Nutrien (NTR)"),
    "AFSI": ("BANKRUPT", "2018-12-01", "AmTrust Financial Services delisted, fraud investigation"),
    "ADSW": ("ACQUIRED_PUBLIC", "2020-10-30", "Advanced Disposal Services acquired by Waste Management (WM)"),
    "ADT": ("YFINANCE_ISSUE", None, "ADT Inc still trades - possible yfinance date range issue"),
    "AESE": ("BANKRUPT", "2024-01-01", "Allied Esports delisted, failed SPAC"),
    "AEZS": ("YFINANCE_ISSUE", None, "Aeterna Zentaris still trades as AEZS on NASDAQ"),
    "AIMC": ("ACQUIRED_PUBLIC", "2019-08-02", "Altra Industrial Motion acquired by Regal Rexnord (RRX)"),
    "ALLK": ("YFINANCE_ISSUE", None, "Allakos still trades as ALLK on NASDAQ"),
    "ALOG": ("ACQUIRED_PRIVATE", "2021-08-18", "Analogic acquired by Altaris Capital Partners"),
    "BKCC": ("YFINANCE_ISSUE", None, "BlackRock Capital Investment Corp - possibly delisted/OTC"),
    "AGR": ("MERGED", "2020-01-01", "Avangrid (AGR) still trades on NYSE"),
    "ZAYO": ("WENT_PRIVATE", "2020-03-09", "Zayo Group taken private by Digital Colony & EQT Partners"),
    "ARNC": ("TICKER_CHANGED", "2020-04-01", "Arconic renamed to Howmet Aerospace (HWM)"),
    "ZUO": ("ACQUIRED_PRIVATE", "2024-06-17", "Zuora acquired by Silver Lake Partners"),
    "ZI": ("ACQUIRED_PRIVATE", "2025-01-28", "ZoomInfo acquired by private equity"),
    "ZIMV": ("YFINANCE_ISSUE", None, "ZimVie spun off from Zimmer Biomet - trades as ZIMV"),
    "ZIOP": ("YFINANCE_ISSUE", None, "ZIOPHARM Oncology renamed to Alaunos Therapeutics (TCRT)"),
    "ZOOM": ("YFINANCE_ISSUE", None, "ZOOM Video is ZM, not ZOOM (data entry error)"),
    "XON": ("YFINANCE_ISSUE", None, "Intrexon rebranded as Precigen (PGEN)"),
    "STWD": ("YFINANCE_ISSUE", None, "Starwood Property Trust still trades as STWD"),
    "STI": ("MERGED", "2019-12-06", "SunTrust merged with BB&T to form Truist (TFC)"),
    "MFAC": ("SPAC_FAILED", "2023-01-01", "Megalith Financial Acquisition SPAC dissolved"),
    "MNK": ("BANKRUPT", "2020-10-12", "Mallinckrodt filed Chapter 11 bankruptcy"),
    "BPMX": ("ACQUIRED_PUBLIC", "2019-06-28", "BioPharmX acquired by Timber Pharmaceuticals"),
    "CNF": ("YFINANCE_ISSUE", None, "CNFinance Holdings - Chinese ADR, possibly delisted"),
    "LMRK": ("ACQUIRED_PUBLIC", "2021-10-01", "Landmark Infrastructure merged into DigitalBridge (DBRG)"),
    "GPP": ("MERGED", "2023-07-01", "Green Plains Partners merged into Green Plains (GPRE)"),
    "CTLT": ("ACQUIRED_PRIVATE", "2024-12-18", "Catalent acquired by Novo Holdings (private)"),
    "PE": ("MERGED", "2021-01-12", "Parsley Energy merged with Pioneer Natural Resources (PXD)"),
    "S": ("MERGED", "2020-04-01", "Sprint merged with T-Mobile (TMUS)"),
    "INFO": ("MERGED", "2022-02-28", "IHS Markit merged with S&P Global (SPGI)"),
    "WBMD": ("ACQUIRED_PRIVATE", "2017-09-15", "WebMD acquired by Internet Brands (KKR portfolio)"),
    "RIG": ("YFINANCE_ISSUE", None, "Transocean still trades as RIG on NYSE"),
    "BFAM": ("WENT_PRIVATE", "2025-01-06", "Bright Horizons still trades - check yfinance"),
    "ESOB": ("ACQUIRED_PUBLIC", "2024-02-14", "Essa Pharma acquired by AstraZeneca (AZN)"),

    # --- Batch 1 research (high trade count tickers) ---
    "NGLS": ("MERGED", "2016-02-17", "Targa Resources Partners merged into Targa Resources Corp (TRGP)"),
    "TUP": ("BANKRUPT", "2024-09-17", "Tupperware Brands filed Chapter 11 bankruptcy; delisted from NYSE"),
    "VGR": ("ACQUIRED_PRIVATE", "2025-05-01", "Vector Group acquired by Japan Tobacco International for $15/share"),
    "DEACU": ("SPAC_FAILED", "2023-01-01", "Dune Acquisition Corporation SPAC; failed to complete business combination"),
    "PARA": ("ACQUIRED_PRIVATE", "2025-07-07", "Paramount Global acquired by Skydance Media (David Ellison)"),
    "APU": ("MERGED", "2019-04-02", "AmeriGas Partners LP merged into parent UGI Corporation (UGI)"),
    "CVET": ("ACQUIRED_PRIVATE", "2022-01-01", "Covetrus acquired by CD&R (Clayton Dubilier & Rice); taken private"),
    "DCP": ("MERGED", "2023-09-15", "DCP Midstream Partners merged into Phillips 66 (PSX)"),
    "RAI": ("ACQUIRED_PUBLIC", "2017-07-25", "Reynolds American acquired by British American Tobacco (BTI) for ~$49.4B"),
    "ACT": ("TICKER_CHANGED", "2014-06-01", "Actavis changed ticker to AGN; later AbbVie (ABBV) acquired Allergan 2020"),
    "LHCG": ("ACQUIRED_PUBLIC", "2023-02-07", "LHC Group acquired by UnitedHealth/Optum (UNH) for ~$5.4B"),
    "TLLP": ("MERGED", "2018-11-08", "Tesoro Logistics LP merged into Andeavor Logistics; later absorbed by MPC/MPLX"),
    "BERY": ("MERGED", "2025-06-01", "Berry Global Group merged with Amcor to form Amcor plc (AMCR)"),
    "CIT": ("MERGED", "2022-01-03", "CIT Group merged into First Citizens BancShares (FCNCA)"),
    "CMCSK": ("TICKER_CHANGED", "2015-09-01", "Comcast CMCSK (Class A Special) reclassified; trades as CMCSA"),
    "GMLP": ("ACQUIRED_PUBLIC", "2021-04-15", "Golar LNG Partners acquired by New Fortress Energy (NFE)"),
    "REVB": ("BANKRUPT", "2022-06-16", "Revlon filed Chapter 11 bankruptcy"),
    "SRLP": ("ACQUIRED_PRIVATE", "2020-01-01", "Sprague Resources LP taken private by Axel Johnson Inc"),
    "SSBK": ("BANKRUPT", "2023-03-12", "Signature Bank seized by FDIC; third-largest US bank failure"),
    "SYRG": ("MERGED", "2017-10-05", "Synergy Resources Corp merged into Centennial Resource Development"),
    "TSS": ("MERGED", "2019-09-18", "Total System Services merged with Global Payments (GPN)"),
    "BPMP": ("MERGED", "2021-03-31", "BP Midstream Partners merged into parent BP plc (BP)"),
    "CDAY": ("TICKER_CHANGED", "2024-02-26", "Ceridian HCM changed name to Dayforce Inc; ticker from CDAY to DAY"),
    "CSLT": ("ACQUIRED_PRIVATE", "2022-06-14", "Castlight Health acquired by Vera Whole Health/Transcarent"),
    "NUAN": ("ACQUIRED_PUBLIC", "2022-03-04", "Nuance Communications acquired by Microsoft (MSFT) for ~$19.7B"),
    "AXLL": ("ACQUIRED_PUBLIC", "2016-08-31", "Axiall Corporation acquired by Westlake Chemical (WLK) for ~$3.8B"),
    "EVBG": ("ACQUIRED_PRIVATE", "2023-02-28", "Everbridge acquired by Thoma Bravo for ~$1.5B"),
    "JIH": ("MERGED", "2021-06-08", "Juniper Industrial Holdings SPAC merged with Janus International (JBI)"),
    "KORS": ("TICKER_CHANGED", "2019-01-02", "Michael Kors renamed to Capri Holdings; ticker changed to CPRI"),
    "PEGI": ("ACQUIRED_PRIVATE", "2020-03-10", "Pattern Energy acquired by Canada Pension Plan Investment Board"),
    "SIRE": ("MERGED", "2019-09-01", "Sisecam Resources LP merged into parent Sisecam Chemicals"),
    "WLL": ("BANKRUPT", "2020-07-01", "Whiting Petroleum filed Chapter 11; later merged into Chord Energy (CHRD)"),
    "CLNC": ("TICKER_CHANGED", "2021-08-01", "Colony Credit Real Estate renamed to Brightspire Capital (BRSP)"),
    "DNKN": ("ACQUIRED_PRIVATE", "2020-12-15", "Dunkin' Brands acquired by Inspire Brands (Roark Capital) for ~$11.3B"),
    "GOGL": ("YFINANCE_ISSUE", None, "Golden Ocean Group still trades as GOGL on NASDAQ"),
    "HCN": ("TICKER_CHANGED", "2018-02-12", "Health Care REIT renamed to Welltower; ticker changed to WELL"),
    "NYCB": ("TICKER_CHANGED", "2024-12-02", "New York Community Bancorp rebranded to Flagstar Financial (FLG)"),
    "PLYA": ("ACQUIRED_PUBLIC", "2025-03-01", "Playa Hotels & Resorts acquired by Hyatt Hotels (H) for ~$2.6B"),
    "WMGI": ("ACQUIRED_PUBLIC", "2020-11-11", "Wright Medical Group acquired by Stryker (SYK) for ~$5.4B"),
    "ANDX": ("MERGED", "2018-10-01", "Andeavor Logistics LP merged into MPLX LP (MPLX)"),
    "DBD": ("BANKRUPT", "2023-06-01", "Diebold Nixdorf filed Chapter 11 bankruptcy; old equity cancelled"),
    "GRUB": ("ACQUIRED_PUBLIC", "2021-06-15", "Grubhub acquired by Just Eat Takeaway.com for ~$7.3B"),
    "MTSC": ("ACQUIRED_PUBLIC", "2021-12-01", "MTS Systems acquired by Amphenol Corporation (APH) for ~$1.7B"),
    "PDCO": ("ACQUIRED_PRIVATE", "2025-01-01", "Patterson Companies acquired by Patient Square Capital; taken private"),
    "PS": ("ACQUIRED_PRIVATE", "2021-04-06", "Pluralsight acquired by Vista Equity Partners for ~$3.5B"),
    "QRTEA": ("TICKER_CHANGED", "2024-01-01", "Qurate Retail Series A tracking stock retired in corporate restructuring"),
    "FLXN": ("ACQUIRED_PUBLIC", "2021-11-19", "Flexion Therapeutics acquired by Pacira BioSciences (PCRX)"),
    "PRAH": ("MERGED", "2021-07-01", "PRA Health Sciences merged with ICON plc (ICLR) in ~$12B deal"),
    "SBNY": ("BANKRUPT", "2023-03-12", "Signature Bank seized by NYDFS/FDIC; deposits acquired by Flagstar"),
    "SMLP": ("YFINANCE_ISSUE", None, "Summit Midstream Partners still trades as SMLP on NYSE"),
    "TPX": ("YFINANCE_ISSUE", None, "Tempur Sealy International still trades as TPX on NYSE"),
    "WAGE": ("ACQUIRED_PUBLIC", "2019-08-30", "WageWorks acquired by HealthEquity (HQY) for ~$2B"),
    "WOOF": ("ACQUIRED_PRIVATE", "2017-09-22", "VCA Inc acquired by Mars Inc for ~$9.1B; taken private"),

    # --- Batch 2 research ---
    "AMED": ("ACQUIRED_PUBLIC", "2024-09-27", "Amedisys acquired by UnitedHealth Group (Optum) for ~$3.3B"),
    "CTRX": ("ACQUIRED_PUBLIC", "2015-07-23", "Catamaran Corporation acquired by UnitedHealth/OptumRx for ~$12.8B"),
    "DPS": ("MERGED", "2018-07-09", "Dr Pepper Snapple merged with Keurig Green Mountain to form KDP"),
    "GAS": ("ACQUIRED_PUBLIC", "2016-07-01", "AGL Resources acquired by Southern Company (SO) for ~$12B"),
    "HRC": ("ACQUIRED_PUBLIC", "2021-12-13", "Hill-Rom Holdings acquired by Baxter International (BAX) for ~$12.5B"),
    "MIK": ("WENT_PRIVATE", "2021-06-22", "Michaels Companies taken private by Apollo Global Management for ~$5B"),
    "MTBC": ("TICKER_CHANGED", "2021-07-01", "MTBC Inc rebranded to CareCloud; ticker changed to CCLD"),
    "NATI": ("ACQUIRED_PUBLIC", "2023-10-02", "National Instruments acquired by Emerson Electric (EMR) for ~$8.2B"),
    "ORCC": ("TICKER_CHANGED", "2023-01-01", "Owl Rock Capital Corp changed ticker to OBDC (Blue Owl Capital)"),
    "QTS": ("WENT_PRIVATE", "2021-09-01", "QTS Realty Trust taken private by Blackstone Infrastructure for ~$10B"),
    "RGC": ("ACQUIRED_PUBLIC", "2018-02-28", "Regal Entertainment acquired by Cineworld Group for ~$5.8B"),
    "SYNH": ("WENT_PRIVATE", "2023-09-14", "Syneos Health taken private by Elliott & Patient Square Capital for ~$7.1B"),
    "VRM": ("BANKRUPT", "2024-11-18", "Vroom wound down ecommerce vehicle operations; ceased business"),
    "AUY": ("ACQUIRED_PUBLIC", "2023-03-31", "Yamana Gold acquired by Pan American Silver (PAAS) and Agnico Eagle (AEM)"),
    "CHNG": ("ACQUIRED_PUBLIC", "2022-10-03", "Change Healthcare acquired by UnitedHealth Group (Optum) for ~$7.8B"),
    "CLNY": ("TICKER_CHANGED", "2022-06-22", "Colony Capital rebranded to DigitalBridge Group; ticker changed to DBRG"),
    "CRY": ("TICKER_CHANGED", "2022-01-03", "CryoLife rebranded to Artivion; ticker changed to AORT"),
    "EV": ("ACQUIRED_PUBLIC", "2021-03-01", "Eaton Vance acquired by Morgan Stanley for ~$7B"),
    "LABL": ("WENT_PRIVATE", "2019-11-01", "Multi-Color Corporation taken private by Platinum Equity for ~$2.5B"),
    "LTD": ("TICKER_CHANGED", "2013-02-01", "Limited Brands changed ticker to LB; later became Bath & Body Works (BBWI)"),
    "ORAN": ("YFINANCE_ISSUE", None, "Orange SA still trades; US ADR may have data retrieval issues"),
    "PACW": ("MERGED", "2023-11-30", "PacWest Bancorp merged with Banc of California; trades as BANC"),
    "PBCT": ("ACQUIRED_PUBLIC", "2022-04-01", "People's United Financial acquired by M&T Bank (MTB) for ~$7.6B"),
    "ROLL": ("TICKER_CHANGED", "2023-01-01", "RBC Bearings ticker changed from ROLL to RBC; still trades"),
    "RXN": ("MERGED", "2022-11-01", "Rexnord water segment merged with Zurn (ZWS); PMC into Regal Rexnord (RRX)"),
    "SEP": ("MERGED", "2019-01-01", "Spectra Energy Partners LP merged into Enbridge (ENB)"),
    "TWKS": ("WENT_PRIVATE", "2025-04-01", "Thoughtworks taken private by Apax Partners for ~$1.75B"),
    "VRTU": ("WENT_PRIVATE", "2021-02-11", "Virtusa Corporation taken private by Baring Private Equity Asia for ~$2B"),
    "WBK": ("ADR_FOREIGN", "2022-09-01", "Westpac Banking delisted US ADR; still trades as WBC.AX on ASX"),
    "BIF": ("YFINANCE_ISSUE", None, "Boulder Growth & Income Fund (closed-end fund); possible data issues"),
    "CAB": ("ACQUIRED_PRIVATE", "2017-09-25", "Cabela's acquired by Bass Pro Shops (private) for ~$5.5B"),
    "DRQ": ("MERGED", "2024-07-03", "Dril-Quip merged with Innovex Downhole Solutions"),
    "GCI": ("MERGED", "2019-11-19", "Old Gannett merged with GateHouse Media/New Media Investment Group"),
    "GFN": ("ACQUIRED_PUBLIC", "2021-06-30", "General Finance Corp acquired by WillScot Mobile Mini (WSC) for ~$1B"),
    "KRFT": ("MERGED", "2015-07-06", "Kraft Foods Group merged with H.J. Heinz to form Kraft Heinz (KHC)"),
    "LM": ("ACQUIRED_PUBLIC", "2020-07-31", "Legg Mason acquired by Franklin Templeton (BEN) for ~$4.5B"),
    "LSXMA": ("MERGED", "2024-09-09", "Liberty SiriusXM tracking stock merged into SiriusXM Holdings (SIRI)"),
    "NTT": ("ADR_FOREIGN", None, "Nippon Telegraph & Telephone; US ADR has data issues"),
    "OAS": ("BANKRUPT", "2020-09-30", "Oasis Petroleum filed Ch.11; emerged then merged into Chord Energy (CHRD)"),
    "POL": ("TICKER_CHANGED", "2020-06-30", "PolyOne Corporation rebranded to Avient; ticker changed to AVNT"),
    "RP": ("WENT_PRIVATE", "2021-04-22", "RealPage taken private by Thoma Bravo for ~$10.2B"),
    "SERV": ("TICKER_CHANGED", "2018-10-01", "ServiceMaster rebranded to Terminix (TMX); later acquired by Rentokil (RTO)"),
    "TELL": ("ACQUIRED_PUBLIC", "2024-10-01", "Tellurian/Driftwood LNG acquired by Woodside Energy (WDS) for ~$900M"),
    "ULTI": ("WENT_PRIVATE", "2019-05-03", "Ultimate Software taken private by Hellman & Friedman for ~$11B; formed UKG"),
    "UMPQ": ("MERGED", "2023-03-01", "Umpqua Holdings merged with Columbia Banking System; trades as COLB"),
    "AY": ("WENT_PRIVATE", "2024-06-17", "Atlantica Sustainable Infrastructure taken private by Energy Capital Partners"),
    "AZEK": ("ACQUIRED_PUBLIC", "2024-10-28", "AZEK Company acquired by James Hardie Industries (JHX) for ~$8.75B"),
    "BKEP": ("ACQUIRED_PRIVATE", "2022-11-01", "Blueknight Energy Partners acquired by Atlas Point Energy (Ergon)"),
    "BRKS": ("TICKER_CHANGED", "2022-09-01", "Brooks Automation rebranded to Azenta Inc; ticker changed to AZTA"),
    "CAJ": ("ADR_FOREIGN", None, "Canon Inc ADR; possible yfinance data retrieval issues"),
    "CPGX": ("MERGED", "2019-12-31", "Enbridge Income Fund Holdings merged into Enbridge (ENB)"),
    "CTRP": ("TICKER_CHANGED", "2019-10-29", "Ctrip.com rebranded to Trip.com Group; ticker changed to TCOM"),
    "EBSB": ("ACQUIRED_PUBLIC", "2022-04-01", "Meridian Bancorp (EBSB) acquired by Independent Bank Corp (INDB)"),
    "ETRN": ("ACQUIRED_PUBLIC", "2024-07-22", "Equitrans Midstream acquired by EQT Corporation in all-stock deal"),
    "FEI": ("ACQUIRED_PUBLIC", "2016-09-19", "FEI Company acquired by Thermo Fisher Scientific (TMO) for ~$4.2B"),
    "HAYN": ("ACQUIRED_PUBLIC", "2024-11-04", "Haynes International acquired by Arconic Corporation"),
    "IBKC": ("MERGED", "2020-11-01", "IBERIABANK merged with First Horizon National Corp (FHN)"),
    "KPLTW": ("SPAC_FAILED", None, "Katapult Holdings warrants (SPAC-related); warrants expired/redeemed"),
    "LLL": ("MERGED", "2019-06-29", "L3 Technologies merged with Harris Corp to form L3Harris (LHX)"),
    "LPT": ("MERGED", "2020-02-04", "Liberty Property Trust merged into Prologis (PLD)"),
    "LSI": ("ACQUIRED_PUBLIC", "2014-05-06", "LSI Corporation acquired by Avago Technologies (now AVGO) for ~$6.6B"),
    "MDSO": ("ACQUIRED_PUBLIC", "2019-10-01", "Medidata Solutions acquired by Dassault Systemes for ~$5.8B"),
    "MIC": ("WENT_PRIVATE", "2022-04-01", "Macquarie Infrastructure Corp internalized/wound down; delisted NYSE"),
    "NYLD": ("TICKER_CHANGED", "2018-08-06", "NRG Yield rebranded to Clearway Energy; ticker changed to CWEN"),
    "OKS": ("MERGED", "2017-06-30", "ONEOK Partners LP merged into parent ONEOK Inc (OKE)"),
    "RLGY": ("TICKER_CHANGED", "2023-04-04", "Realogy Holdings rebranded to Anywhere Real Estate; ticker to HOUS"),
    "RNWK": ("YFINANCE_ISSUE", None, "RealNetworks still exists but micro-cap; possible data issues"),

    # --- Batch 3 research ---
    "ALTR": ("ACQUIRED_PUBLIC", "2024-12-19", "Altair Engineering acquired by Siemens AG for ~$10.6B"),
    "AMEH": ("TICKER_CHANGED", "2023-06-01", "Apollo Medical/ApolloMed rebranded to Astrana Health; ticker to ASTH"),
    "BRCM": ("ACQUIRED_PUBLIC", "2016-02-01", "Broadcom Corp acquired by Avago Technologies; combined as AVGO"),
    "BRP": ("TICKER_CHANGED", "2023-06-05", "BRP Group rebranded to Baldwin Insurance Group; ticker to BWIN"),
    "CCXI": ("ACQUIRED_PUBLIC", "2022-10-18", "ChemoCentryx acquired by Amgen for ~$3.7B"),
    "CMLFU": ("SPAC_FAILED", "2023-01-01", "CM Life Sciences III SPAC; failed to complete merger; liquidated"),
    "FII": ("TICKER_CHANGED", "2020-01-01", "Federated Investors rebranded to Federated Hermes; ticker to FHI"),
    "HTA": ("MERGED", "2022-07-20", "Healthcare Trust of America merged with Healthcare Realty Trust (HR)"),
    "INST": ("ACQUIRED_PRIVATE", "2024-03-21", "Instructure Holdings taken private by KKR for ~$4.8B"),
    "JCOM": ("TICKER_CHANGED", "2022-07-01", "j2 Global renamed to Ziff Davis; ticker changed to ZD"),
    "LANC": ("YFINANCE_ISSUE", None, "Lancaster Colony still trades as LANC on NASDAQ"),
    "MDC": ("ACQUIRED_PUBLIC", "2024-02-02", "M.D.C. Holdings acquired by Sekisui House for ~$4.9B"),
    "NUVA": ("MERGED", "2023-09-01", "NuVasive merged with Globus Medical; trades as GMED"),
    "PRSP": ("ACQUIRED_PRIVATE", "2022-04-14", "Perspecta acquired by Peraton (Veritas Capital) for ~$7.1B"),
    "RAVN": ("ACQUIRED_PUBLIC", "2021-08-23", "Raven Industries acquired by CNH Industrial for ~$2.1B"),
    "RBS": ("TICKER_CHANGED", "2020-07-22", "Royal Bank of Scotland renamed NatWest Group; ticker to NWG"),
    "SI": ("BANKRUPT", "2023-03-08", "Silvergate Capital announced voluntary liquidation amid crypto crisis"),
    "SNH": ("TICKER_CHANGED", "2020-02-28", "Senior Housing Properties Trust renamed to Diversified Healthcare Trust (DHC)"),
    "TGE": ("ACQUIRED_PRIVATE", "2019-11-15", "Tallgrass Energy taken private by Blackstone Infrastructure for ~$5.1B"),
    "VIVO": ("ACQUIRED_PRIVATE", "2023-02-27", "Meridian Bioscience acquired by SD Biosensor/SJL Partners for ~$1.53B"),
}


def get_failed_tickers():
    """Return set of tickers that only have FAILED entries in the log."""
    failed = set()
    success = set()
    with open(LOG_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row['Ticker'].strip()
            status = row['Status'].strip()
            if status == 'SUCCESS':
                success.add(ticker)
            elif status == 'FAILED':
                failed.add(ticker)
    return sorted(failed - success)


def get_trade_counts(tickers):
    """Return dict of ticker -> trade count from source data."""
    df = pl.read_parquet(SOURCE_FILE)
    df = df.with_columns(pl.col("Ticker").cast(pl.Utf8).fill_null("N/A"))
    df = df.filter(pl.col("Ticker").is_in(tickers))
    counts = df.group_by("Ticker").len()
    return {row[0]: row[1] for row in counts.iter_rows()}


def load_professor_data():
    """Load professor's delisted ticker research."""
    data = {}
    if not os.path.exists(PROFESSOR_FILE):
        return data
    with open(PROFESSOR_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get("original_ticker", "").strip()
            if ticker:
                data[ticker] = row
    return data


def infer_category(ticker):
    """Infer category from ticker pattern when no other data exists."""
    # Data errors - clearly not real tickers
    if " " in ticker or len(ticker) > 8:
        return "DATA_ERROR", None, f"Invalid ticker format: '{ticker}'"

    # Index symbols
    if ticker.startswith("^"):
        return "DATA_ERROR", None, "Index symbol, not a stock"

    # International exchange suffixes
    if "." in ticker:
        parts = ticker.split(".")
        suffix = parts[-1]
        if suffix in ("L", "TO", "AX", "PA", "MI", "F", "SW", "AS", "ST",
                       "TI", "MU", "HK", "T", "SS", "SZ", "WI", "VI"):
            return "ADR_FOREIGN", None, f"International exchange listing (.{suffix})"

    # ADR patterns - tickers ending in Y or F that are 5+ chars
    if len(ticker) >= 5:
        if ticker.endswith("Y") and not ticker.endswith("RY"):
            return "ADR_FOREIGN", None, "Likely foreign ADR (ends in Y, 5+ chars)"
        if ticker.endswith("F") and ticker not in ("SCHF", "CHEF"):
            return "ADR_FOREIGN", None, "Likely foreign OTC (ends in F, 5+ chars)"

    # Preferred shares with $ sign
    if "$" in ticker:
        return "YFINANCE_ISSUE", None, "Preferred share notation - may need format fix"

    return "UNKNOWN", None, "Insufficient information to categorize"


def categorize_from_professor(row):
    """Extract category from professor's research data."""
    status = row.get("status", "").strip().upper()
    reason = row.get("reason", "").strip().upper()
    notes = row.get("notes", "").strip()
    new_ticker = row.get("new_ticker", "").strip()

    # Extract date from notes if possible
    date = None
    import re
    date_match = re.search(r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})', notes)
    if date_match:
        date = date_match.group(1)
    # Also try "Month YYYY" or "YYYY" patterns
    if not date:
        year_match = re.search(r'\b(20\d{2})\b', notes)
        if year_match:
            date = year_match.group(1)

    if reason == "BANKRUPT" or "BANKRUPT" in reason:
        return "BANKRUPT", date, notes
    if reason == "WENT_PRIVATE" or "PRIVATE" in reason:
        return "WENT_PRIVATE", date, notes
    if reason == "ACQUIRED":
        acquiring = row.get("acquiring_company", "").strip()
        if new_ticker and new_ticker.lower() not in ("private", "unknown", "none", ""):
            return "ACQUIRED_PUBLIC", date, f"Acquired by {acquiring}. {notes}".strip()
        else:
            return "ACQUIRED_PRIVATE", date, f"Acquired by {acquiring}. {notes}".strip()
    if reason == "MERGED" or "MERGE" in reason:
        return "MERGED", date, notes
    if reason == "COMPANY_RENAME" or reason == "NO_CHANGE":
        return "TICKER_CHANGED", date, notes
    if "SPAC" in reason or "SPAC" in status:
        return "SPAC_FAILED", date, notes
    if reason == "DATA_ERROR" or "ERROR" in reason:
        return "DATA_ERROR", date, notes
    if reason == "LIQUIDATED":
        return "BANKRUPT", date, f"Liquidated. {notes}"
    if reason == "FRAUD":
        return "BANKRUPT", date, f"Fraud/delisted. {notes}"
    if status == "NOT_FOUND":
        return "UNKNOWN", date, notes
    if reason == "DELISTED":
        return "UNKNOWN", date, f"Delisted (reason unclear). {notes}"

    return "UNKNOWN", date, notes


def main():
    os.chdir(PROJECT_ROOT)

    failed_tickers = get_failed_tickers()
    trade_counts = get_trade_counts(failed_tickers)
    professor_data = load_professor_data()

    print(f"Failed tickers: {len(failed_tickers)}")
    print(f"Professor matches: {sum(1 for t in failed_tickers if t in professor_data)}")
    print(f"Manual research matches: {sum(1 for t in failed_tickers if t in MANUAL_RESEARCH)}")

    rows = []
    for ticker in failed_tickers:
        trades = trade_counts.get(ticker, 0)

        # Priority: 1) Manual research, 2) Professor data, 3) Pattern inference
        if ticker in MANUAL_RESEARCH:
            cat, date, notes = MANUAL_RESEARCH[ticker]
        elif ticker in professor_data:
            cat, date, notes = categorize_from_professor(professor_data[ticker])
        else:
            cat, date, notes = infer_category(ticker)

        rows.append({
            "ticker": ticker,
            "category": cat,
            "date": date or "",
            "trade_count": trades,
            "notes": notes or "",
        })

    # Sort by trade count descending
    rows.sort(key=lambda r: -r["trade_count"])

    # Write output
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "category", "date", "trade_count", "notes"])
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    from collections import Counter
    cats = Counter(r["category"] for r in rows)
    print(f"\nCategory breakdown:")
    for cat, count in cats.most_common():
        trades_in_cat = sum(r["trade_count"] for r in rows if r["category"] == cat)
        print(f"  {cat:20s}: {count:4d} tickers, {trades_in_cat:5d} trades")

    print(f"\nTotal: {len(rows)} tickers, {sum(r['trade_count'] for r in rows)} trades")
    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
