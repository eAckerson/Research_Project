#!/usr/bin/env python3
"""
ONOS_CORE_FEATURE_EXTRACTOR.py
SDN feature extraction with deep ONOS integration for enhanced UNSW-NB15 features
Leverages ONOS global network view + Argus + nDPI for comprehensive feature extraction
Author: ACKERSON Emmanuel
"""

import os, time, yaml, logging, subprocess, threading, csv, signal, shutil, re, math, queue
from datetime import datetime, timedelta
from collections import deque, defaultdict
import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
import warnings
import pickle
import tensorflow as tf

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# ============================================
# LOGGING SETUP - MUST BE BEFORE IDS INTEGRATION
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ONOS_Extractor")


# ============================================
# CONFIGURATION
# ============================================
def load_config(path="config/paths.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

cfg = load_config()
ONOS_IP   = cfg["onos"]["ip"]
ONOS_PORT = cfg["onos"]["port"]
ONOS_USER = cfg["onos"]["user"]
ONOS_PASS = cfg["onos"]["pass"]


MON_INTERFACE = cfg["monitor"].get("interface", "s3-eth1")

# Windows monitoring server
WINDOWS_SERVER = "http://196.47.197.1:8000"
PREDICTIONS_ENDPOINT = f"{WINDOWS_SERVER}/predictions/batch"
FEATURES_ENDPOINT = f"{WINDOWS_SERVER}/features/batch"

# HTTP transmission settings
HTTP_TIMEOUT = 5
HTTP_BATCH_SIZE = 32
IDS_BATCH_SIZE = 32
IDS_BATCH_WAIT_SECONDS = 0.1
IDS_MAX_RECORD_AGE_SECONDS = 30
TMP_ARGUS_OUTPUT = "/tmp/flows_argus.csv"
TMP_NDPI_OUTPUT  = "/tmp/flows_ndpi.csv"


os.makedirs("outputs", exist_ok=True)

# ============================================
# GRACEFUL STOP
# ============================================
running = True
def signal_handler(sig, frame):
    global running
    logger.info("Shutting down...")
    running = False
signal.signal(signal.SIGINT, signal_handler)

# ============================================
# ENHANCED UNSW-NB15 FEATURES WITH ONOS INTEGRATION
# ============================================
ENHANCED_UNSW_FEATURES = [
    # Core 5-tuple + state
    "saddr", "sport", "daddr", "dport", "proto", "state",
    
    # Flow duration and sizes
    "dur", "sbytes", "dbytes", 
    
    # TTL values
    "sttl", "dttl",
    
    # Loss metrics
    "sloss", "dloss",
    
    # Service (from nDPI)
    "service",
    
    # Load rates
    "sload", "dload",

    # Packet counts
    "spkts", "dpkts",
    
    # TCP windows
    "swin", "dwin",
    
    # TCP base sequences
    "stcpb", "dtcpb",
    
    # Mean sizes
    "smeansz", "dmeansz",
    
    # Jitter
    "sjit", "djit",
    
    # Inter-packet times
    "sintpkt", "dintpkt",
    
    # TCP timing
    "tcprtt", "synack", "ackdat",
    
    # Behavioral
    "is_sm_ips_ports",
    
    # Connection tracking (UNSW-NB15)
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    
    # ONOS-ENHANCED FEATURES (Key Contribution)
    "onos_flow_priority",
    "onos_table_id",
    "onos_flow_duration",
    "onos_packet_count",
    "onos_byte_count",
    "onos_switch_id",
    "onos_ingress_port",
    "onos_flow_actions",
    "onos_path_length",
    "onos_flow_age",
    
    # Metadata
    "timestamp"
]

# ============================================
# IDS FEATURE BUFFER
# ============================================

MODEL_FEATURES = [
    'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl',
    'sloss', 'dloss', 'sload', 'dload', 'spkts', 'dpkts',
    'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz',
    'sjit', 'djit', 'sintpkt', 'dintpkt', 'tcprtt',
    'synack', 'ackdat', 'is_sm_ips_ports',
    'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm',
    'ct_src_ltm', 'ct_src_dport_ltm',
    'ct_dst_sport_ltm', 'ct_dst_src_ltm'
]

# Temporary in-memory buffer for CNN input records
feature_queue = queue.Queue()

logger.info(
    f"IDS feature buffer initialized: {len(MODEL_FEATURES)} model features"
)

# ============================================
# ADVANCED ONOS CLIENT WITH SDN INTELLIGENCE
# ============================================
class ONOSIntelligenceEngine:
    def __init__(self, ip, port, user, passwd):
        self.base_url = f"http://{ip}:{port}/onos/v1"
        self.auth = (user, passwd)
        self.flow_cache = {}
        self.topology_cache = {}
        self.host_cache = {}
        self.device_cache = {}
        
    def get_json(self, endpoint):
        try:
            response = requests.get(
                f"{self.base_url}/{endpoint}",
                auth=self.auth,
                timeout=15
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.debug(f"ONOS API {endpoint}: {response.status_code}")
                return None
        except Exception as e:
            logger.debug(f"ONOS connection failed: {e}")
            return None
    
    def refresh_global_view(self):
        """Refresh ONOS global network intelligence"""
        try:
            self.flow_cache = self._get_enhanced_flows()
            self.topology_cache = self.get_json("topology")
            self.host_cache = self.get_json("hosts") or {"hosts": []}
            self.device_cache = self.get_json("devices") or {"devices": []}
            
            logger.info(f"ONOS Global View: {len(self.flow_cache)} flows, "
                       f"{len(self.host_cache.get('hosts', []))} hosts, "
                       f"{len(self.device_cache.get('devices', []))} devices")
            
        except Exception as e:
            logger.warning(f"ONOS global view refresh failed: {e}")
    
    def _get_enhanced_flows(self):
        """Get flows with detailed statistics"""
        flows_data = self.get_json("flows")
        flow_stats = self.get_json("statistics/flows")
        
        enhanced_flows = {}
        
        if flows_data and "flows" in flows_data:
            for flow in flows_data["flows"]:
                flow_id = flow.get("id", "unknown")
                device_id = flow.get("deviceId", "unknown")
                
                enhanced_flows[flow_id] = {
                    "deviceId": device_id,
                    "priority": flow.get("priority", 0),
                    "tableId": flow.get("tableId", 0),
                    "timeout": flow.get("timeout", 0),
                    "duration": flow.get("durationNanos", 0) / 1e9,
                    "packetCount": 0,
                    "byteCount": 0,
                    "treatment": flow.get("treatment", {}),
                    "selector": flow.get("selector", {}),
                    "ingressPort": self._extract_ingress_port(flow),
                    "egressPorts": self._extract_egress_ports(flow),
                    "installTime": datetime.now().timestamp() - (flow.get("durationNanos", 0) / 1e9)
                }
        
        if flow_stats and "flows" in flow_stats:
            for stat_flow in flow_stats["flows"]:
                flow_id = stat_flow.get("flowId")
                if flow_id in enhanced_flows:
                    enhanced_flows[flow_id].update({
                        "packetCount": stat_flow.get("packets", 0),
                        "byteCount": stat_flow.get("bytes", 0)
                    })
        
        return enhanced_flows
    
    def _extract_ingress_port(self, flow):
        criteria = flow.get("selector", {}).get("criteria", [])
        for criterion in criteria:
            if criterion.get("type") == "IN_PORT":
                return criterion.get("port", 0)
        return 0
    
    def _extract_egress_ports(self, flow):
        ports = []
        instructions = flow.get("treatment", {}).get("instructions", [])
        for instruction in instructions:
            if instruction.get("type") == "OUTPUT":
                ports.append(instruction.get("port", 0))
        return ports
    
    def correlate_flow_with_onos(self, argus_flow):
        onos_features = {
            "onos_flow_priority": 0,
            "onos_table_id": 0,
            "onos_flow_duration": 0,
            "onos_packet_count": 0,
            "onos_byte_count": 0,
            "onos_switch_id": "unknown",
            "onos_ingress_port": 0,
            "onos_flow_actions": 0,
            "onos_path_length": 1,
            "onos_flow_age": 0
        }
        
        matching_flow = self._find_matching_onos_flow(argus_flow)
        
        if matching_flow:
            onos_features.update({
                "onos_flow_priority": matching_flow.get("priority", 0),
                "onos_table_id": matching_flow.get("tableId", 0),
                "onos_flow_duration": matching_flow.get("duration", 0),
                "onos_packet_count": matching_flow.get("packetCount", 0),
                "onos_byte_count": matching_flow.get("byteCount", 0),
                "onos_switch_id": matching_flow.get("deviceId", "unknown"),
                "onos_ingress_port": matching_flow.get("ingressPort", 0),
                "onos_flow_actions": len(matching_flow.get("treatment", {}).get("instructions", [])),
                "onos_flow_age": matching_flow.get("installTime", 0)
            })
            
            onos_features["onos_path_length"] = self._calculate_path_length(
                argus_flow.get("saddr"), 
                argus_flow.get("daddr")
            )
        
        return onos_features
    
    def _find_matching_onos_flow(self, argus_flow):
        src_ip = argus_flow.get("saddr")
        dst_ip = argus_flow.get("daddr")
        proto = argus_flow.get("proto")
        src_port = argus_flow.get("sport")
        dst_port = argus_flow.get("dport")
        
        for flow_id, onos_flow in self.flow_cache.items():
            if self._flows_match(onos_flow, src_ip, dst_ip, proto, src_port, dst_port):
                return onos_flow
        
        return None
    
    def _flows_match(self, onos_flow, src_ip, dst_ip, proto, src_port, dst_port):
        selector = onos_flow.get("selector", {})
        criteria = selector.get("criteria", [])
        
        match_score = 0
        
        for criterion in criteria:
            criterion_type = criterion.get("type")
            
            if criterion_type == "IPV4_SRC" and criterion.get("ip") == src_ip:
                match_score += 1
            elif criterion_type == "IPV4_DST" and criterion.get("ip") == dst_ip:
                match_score += 1
            elif criterion_type == "IP_PROTO" and str(criterion.get("protocol")) == str(proto):
                match_score += 1
            elif criterion_type == "TCP_SRC" and src_port and criterion.get("tcpPort") == int(src_port):
                match_score += 1
            elif criterion_type == "TCP_DST" and dst_port and criterion.get("tcpPort") == int(dst_port):
                match_score += 1
            elif criterion_type == "UDP_SRC" and src_port and criterion.get("udpPort") == int(src_port):
                match_score += 1
            elif criterion_type == "UDP_DST" and dst_port and criterion.get("udpPort") == int(dst_port):
                match_score += 1
        
        return match_score >= 2
    
    def _calculate_path_length(self, src_ip, dst_ip):
        try:
            src_host = self._find_host_for_ip(src_ip)
            dst_host = self._find_host_for_ip(dst_ip)
            
            if src_host and dst_host:
                return 3
            else:
                return 1
        
        except Exception:
            return 1
    
    def _find_host_for_ip(self, ip):
        for host in self.host_cache.get("hosts", []):
            if any(ip == loc.get("ip") for loc in host.get("locations", [])):
                return host
        return None

# ============================================
# CAPTURE FUNCTIONS
# ============================================
def check_dependencies():
    tools = ["argus", "ra", "tcpdump", "timeout"]
    missing = [t for t in tools if not shutil.which(t)]
    if missing:
        logger.error(f"Missing tools: {missing}")
        return False
    return True

def capture_flows(duration=5):
    """Capture packets and convert to flows using Argus"""
    tmp_pcap = "/tmp/capture.pcap"
    
    for f in [tmp_pcap, TMP_ARGUS_OUTPUT]:
        if os.path.exists(f):
            os.remove(f)
    
    logger.info(f"Capturing on {MON_INTERFACE} for {duration}s")
    
    tcpdump_cmd = [
        "sudo", "timeout", str(duration),
        "tcpdump", "-i", MON_INTERFACE, "-n", 
        "-w", tmp_pcap, "ip"
    ]
    
    result = subprocess.run(tcpdump_cmd, capture_output=True)
    
    if not os.path.exists(tmp_pcap) or os.path.getsize(tmp_pcap) < 100:
        logger.warning("No packets captured")
        return False
    
    argus_cmd = (
        f"argus -r {tmp_pcap} -w - | "
        f"ra -L - -c ',' -s saddr sport daddr dport proto state dur "
        f"sbytes dbytes sttl dttl sloss dloss spkts dpkts swin dwin "
        f"stcpb dtcpb > {TMP_ARGUS_OUTPUT}"
    )
    subprocess.run(argus_cmd, shell=True)
    
    if os.path.exists(TMP_ARGUS_OUTPUT) and os.path.getsize(TMP_ARGUS_OUTPUT) > 50:
        logger.info("Argus processing successful")
        return True
    
    return False

def process_ndpi():
    """Process with nDPI for service classification"""
    tmp_pcap = "/tmp/capture.pcap"
    if not os.path.exists(tmp_pcap):
        return False
    
    if os.path.exists(TMP_NDPI_OUTPUT):
        os.remove(TMP_NDPI_OUTPUT)
    
    ndpi_cmd = f"ndpiReader -i {tmp_pcap} -C {TMP_NDPI_OUTPUT} -q"
    result = subprocess.run(ndpi_cmd, shell=True, capture_output=True)
    
    return os.path.exists(TMP_NDPI_OUTPUT)

# ============================================
# DATA PROCESSING
# ============================================
def parse_argus_data():
    """Parse Argus output with UNSW-NB15 column names"""
    if not os.path.exists(TMP_ARGUS_OUTPUT):
        return None
    
    try:
        df = pd.read_csv(TMP_ARGUS_OUTPUT, comment='#', dtype=str)
        if df.empty:
            return None
        
        df.columns = [col.strip().lower() for col in df.columns]
        
        unsw_column_map = {
            'srcaddr': 'saddr', 'saddr': 'saddr',
            'dstaddr': 'daddr', 'daddr': 'daddr',
            'srcport': 'sport', 'sport': 'sport',
            'dstport': 'dport', 'dport': 'dport',
            'protocol': 'proto', 'proto': 'proto',
            'state': 'state',
            'duration': 'dur', 'dur': 'dur',
            'srcbytes': 'sbytes', 'sbytes': 'sbytes',
            'dstbytes': 'dbytes', 'dbytes': 'dbytes',
            'srcttl': 'sttl', 'sttl': 'sttl',
            'dstttl': 'dttl', 'dttl': 'dttl',
            'srcloss': 'sloss', 'sloss': 'sloss',
            'dstloss': 'dloss', 'dloss': 'dloss',
            'srcpkts': 'spkts', 'spkts': 'spkts',
            'dstpkts': 'dpkts', 'dpkts': 'dpkts',
            'srcwin': 'swin', 'swin': 'swin',
            'dstwin': 'dwin', 'dwin': 'dwin',
            'srctcpb': 'stcpb', 'stcpb': 'stcpb',
            'dsttcpb': 'dtcpb', 'dtcpb': 'dtcpb'
        }
        
        for old_name, new_name in unsw_column_map.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        for col in ENHANCED_UNSW_FEATURES:
            if col not in df.columns and not col.startswith('onos_') and col != 'timestamp':
                df[col] = 0
        
        numeric_cols = ['sport', 'dport', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl',
                       'sloss', 'dloss', 'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
        
    except Exception as e:
        logger.error(f"Argus parsing error: {e}")
        return None

def parse_ndpi_data():
    """Parse nDPI output for service information"""
    service_map = {}
    
    if not os.path.exists(TMP_NDPI_OUTPUT):
        return service_map
    
    try:
        with open(TMP_NDPI_OUTPUT, 'r') as f:
            for line in f:
                if '|' in line and not line.startswith('#'):
                    parts = line.strip().split('|')
                    if len(parts) >= 11:
                        src_ip, src_port = parts[5], parts[6]
                        dst_ip, dst_port = parts[7], parts[8]
                        service = parts[10].strip()
                        
                        key1 = (src_ip, src_port, dst_ip, dst_port)
                        key2 = (dst_ip, dst_port, src_ip, src_port)
                        
                        service_map[key1] = service
                        service_map[key2] = service
                        
    except Exception as e:
        logger.warning(f"nDPI parsing error: {e}")
    
    return service_map

def calculate_unsw_features(df):
    """Calculate UNSW-NB15 derived features"""
    if df is None or df.empty:
        return None
    
    df['smeansz'] = np.where(df['spkts'] > 0, df['sbytes'] / df['spkts'], 0)
    df['dmeansz'] = np.where(df['dpkts'] > 0, df['dbytes'] / df['dpkts'], 0)
    
    df['sload'] = np.where(df['dur'] > 0, df['sbytes'] / df['dur'], 0)
    df['dload'] = np.where(df['dur'] > 0, df['dbytes'] / df['dur'], 0)
    
    df['sintpkt'] = np.where(df['spkts'] > 0, df['dur'] / df['spkts'], 0)
    df['dintpkt'] = np.where(df['dpkts'] > 0, df['dur'] / df['dpkts'], 0)
    
    df['sjit'] = 0
    df['djit'] = 0
    df['tcprtt'] = 0
    df['synack'] = 0
    df['ackdat'] = 0
    
    df['is_sm_ips_ports'] = (df['saddr'] == df['daddr']) & (df['sport'] == df['dport'])
    
    return df

# ============================================
# CONNECTION TRACKING
# ============================================
class UNSWConnectionTracker:
    def __init__(self, window_seconds=30):
        self.window = timedelta(seconds=window_seconds)
        self.flows = deque()
    
    def add_flow(self, flow):
        flow_entry = {
            'timestamp': datetime.now(),
            'saddr': flow.get('saddr'),
            'daddr': flow.get('daddr'),
            'sport': flow.get('sport'),
            'dport': flow.get('dport'),
            'proto': flow.get('proto'),
            'service': flow.get('service')
        }
        self.flows.append(flow_entry)
        self._clean_old_flows()
    
    def _clean_old_flows(self):
        now = datetime.now()
        while self.flows and (now - self.flows[0]['timestamp']) > self.window:
            self.flows.popleft()
    
    def count_connections(self, filters):
        count = 0
        for flow in self.flows:
            match = True
            for key, value in filters.items():
                if value is not None and flow.get(key) != value:
                    match = False
                    break
            if match:
                count += 1
        return count

def add_connection_tracking_features(df, tracker):
    """Add UNSW-NB15 connection tracking features"""
    if df is None or df.empty:
        return df
    
    ct_features = [
        'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
        'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm'
    ]
    
    for feature in ct_features:
        df[feature] = 0
    
    for idx, row in df.iterrows():
        saddr = row.get('saddr')
        daddr = row.get('daddr')
        sport = row.get('sport')
        dport = row.get('dport')
        proto = row.get('proto')
        service = row.get('service')
        
        df.at[idx, 'ct_srv_src'] = tracker.count_connections({'service': service, 'saddr': saddr})
        df.at[idx, 'ct_srv_dst'] = tracker.count_connections({'service': service, 'daddr': daddr})
        df.at[idx, 'ct_dst_ltm'] = tracker.count_connections({'daddr': daddr})
        df.at[idx, 'ct_src_ltm'] = tracker.count_connections({'saddr': saddr})
        df.at[idx, 'ct_src_dport_ltm'] = tracker.count_connections({'saddr': saddr, 'dport': dport})
        df.at[idx, 'ct_dst_sport_ltm'] = tracker.count_connections({'daddr': daddr, 'sport': sport})
        df.at[idx, 'ct_dst_src_ltm'] = tracker.count_connections({'saddr': saddr, 'daddr': daddr})
        
        tracker.add_flow(row.to_dict())
    
    return df



# ============================================
# ADD FEATURES TO IDS BUFFER
# ============================================

def enqueue_model_features(df):
    """
    Submit AnDO flow features for live local CNN inference.

    If the bounded queue is full, discard the oldest record
    to prioritise recently extracted flows.
    """

    if not running:
        logger.warning("CNN queue submission rejected: shutdown in progress")
        return 0

    if df is None or df.empty:
        return 0

    missing_features = [
        feature for feature in MODEL_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        logger.error(
            f"Cannot submit flows. Missing CNN features: {missing_features}"
        )
        return 0

    model_df = df[MODEL_FEATURES].copy()

    # Replace invalid numerical values before preprocessing.
    model_df = model_df.replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

    records_added = 0
    records_dropped = 0

    for record in model_df.to_dict(orient="records"):

        if not running:
            break

        clean_record = {}

        for feature, value in record.items():
            if isinstance(value, np.generic):
                value = value.item()

            clean_record[feature] = value

        # Internal timestamp; not a CNN input feature.
        clean_record["_queued_at"] = time.monotonic()

        # Discard the oldest record when the queue is full.
        if feature_queue.full():
            try:
                feature_queue.get_nowait()
                feature_queue.task_done()
                records_dropped += 1
            except queue.Empty:
                pass

        try:
            feature_queue.put_nowait(clean_record)
            records_added += 1
        except queue.Full:
            records_dropped += 1

    logger.info(
        "CNN queue update: submitted=%d, dropped_oldest=%d, queue_size=%d",
        records_added,
        records_dropped,
        feature_queue.qsize()
    )

    return records_added



# ============================================
# LOCAL CNN INFERENCE WORKER
# ============================================

CNN_ARTIFACT_DIR = "preprocessing_artifacts"
CNN_MODEL_PATH = "cnn_multiK_model_best.h5"


def load_cnn_artifacts():
    """
    Load the CNN model and preprocessing artifacts once.
    """

    def load_pickle(filename):
        path = f"{CNN_ARTIFACT_DIR}/{filename}.pkl"

        with open(path, "rb") as file:
            return pickle.load(file)

    feature_list = load_pickle("feature_columns")
    scaler = load_pickle("scaler")
    proto_map = load_pickle("proto_smooth_map")
    state_map = load_pickle("state_smooth_map")
    global_mean = load_pickle("global_mean")
    classes = load_pickle("label_encoder_classes")

    if len(feature_list) != 34:
        raise ValueError(
            f"Expected 34 features, found {len(feature_list)}"
        )

    model = tf.keras.models.load_model(CNN_MODEL_PATH)

    if model.input_shape[1:] != (1, 34, 1):
        raise ValueError(
            f"Unexpected CNN input shape: {model.input_shape}"
        )

    if model.output_shape[-1] != len(classes):
        raise ValueError(
            "CNN output dimension does not match class labels."
        )

    logger.info("Local CNN model loaded: %s", CNN_MODEL_PATH)
    logger.info("CNN input shape: %s", model.input_shape)
    logger.info("CNN output shape: %s", model.output_shape)

    return (
        model,
        feature_list,
        scaler,
        proto_map,
        state_map,
        global_mean,
        classes
    )


def preprocess_cnn_records(
    records,
    feature_list,
    scaler,
    proto_map,
    state_map,
    global_mean
):
    """
    Apply the same preprocessing used by the validated
    standalone local CNN inference implementation.
    """

    if not records:
        raise ValueError("No feature records received.")

    processed_rows = []

    for record in records:

        row = record.copy()

        row["proto"] = proto_map.get(
            row.get("proto"), global_mean
        )

        row["state"] = state_map.get(
            row.get("state"), global_mean
        )

        processed_rows.append([
            row.get(feature, 0.0)
            for feature in feature_list
        ])

    X = np.asarray(processed_rows, dtype=np.float64)

    X_scaled = scaler.transform(X)

    X_scaled = X_scaled.reshape(
        len(records), 1, 34, 1
    ).astype(np.float32)

    if not np.isfinite(X_scaled).all():
        raise ValueError(
            "Non-finite values detected in CNN input."
        )

    return X_scaled


def ids_batch_sender():
    """
    Consume recent AnDO feature records and perform local CNN inference.

    Records exceeding the maximum queue age are discarded.
    Pending records are discarded during shutdown.
    """

    global running

    logger.info("Local CNN inference thread starting")

    try:
        (
            model,
            feature_list,
            scaler,
            proto_map,
            state_map,
            global_mean,
            classes
        ) = load_cnn_artifacts()

        # Warm up the model once.
        warmup_record = {
            feature: 0.0 for feature in feature_list
        }
        warmup_record["proto"] = "tcp"
        warmup_record["state"] = "FIN"

        warmup_input = preprocess_cnn_records(
            [warmup_record] * IDS_BATCH_SIZE,
            feature_list,
            scaler,
            proto_map,
            state_map,
            global_mean
        )

        model(warmup_input, training=False)

        logger.info("Local CNN warm-up completed")

    except Exception:
        logger.exception("Local CNN initialisation failed")
        running = False
        return

    try:

        while running:

            batch = []

            # Wait briefly for the first record.
            try:
                first_record = feature_queue.get(timeout=0.1)
                batch.append(first_record)

            except queue.Empty:
                continue

            # Collect additional records for at most 0.1 seconds.
            deadline = (
                time.monotonic() + IDS_BATCH_WAIT_SECONDS
            )

            while running and len(batch) < IDS_BATCH_SIZE:

                remaining_time = deadline - time.monotonic()

                if remaining_time <= 0:
                    break

                try:
                    record = feature_queue.get(
                        timeout=remaining_time
                    )
                    batch.append(record)

                except queue.Empty:
                    break

            # If shutdown has started, discard the current batch.
            if not running:
                for _ in batch:
                    feature_queue.task_done()
                break

            # Reject records that have waited too long.
            now = time.monotonic()

            fresh_records = [
                record for record in batch
                if now - record.get("_queued_at", now)
                <= IDS_MAX_RECORD_AGE_SECONDS
            ]

            stale_count = len(batch) - len(fresh_records)

            if stale_count:
                logger.warning(
                    "Discarded %d stale CNN records",
                    stale_count
                )

            try:

                if not fresh_records:
                    continue

                # Do not start a new inference batch after shutdown.
                if not running:
                    break

                start_time = time.perf_counter()

                X_batch = preprocess_cnn_records(
                    fresh_records,
                    feature_list,
                    scaler,
                    proto_map,
                    state_map,
                    global_mean
                )

                preprocessing_time = (
                    time.perf_counter() - start_time
                )

                inference_start = time.perf_counter()

                probabilities_batch = model(
                    X_batch,
                    training=False
                ).numpy()

                inference_time = (
                    time.perf_counter() - inference_start
                )

#....................................................................................
   
#.........................................................................................
                prediction_records = []

                for index, probabilities in enumerate(
                    probabilities_batch, start=1
                ):

                    class_id = int(np.argmax(probabilities))
                    confidence = float(probabilities[class_id])
                    label = str(classes[class_id])

                    prediction_record = {
                        "timestamp": datetime.now().isoformat(),
                        "class_id": class_id,
                        "label": label,
                        "confidence": round(confidence, 4)
                    }

                    prediction_records.append(prediction_record)

                   # logger.info(
                     #   "CNN prediction [%d/%d]: "
                     #   "class_id=%d, label=%s, confidence=%.4f",
                     #   index,
                     #   len(probabilities_batch),
                      #  class_id,
                     #   label,
                    #    confidence
                   # )

                # Send completed prediction batch to Windows.
                if prediction_records:
                    try:
                        response = requests.post(
                            PREDICTIONS_ENDPOINT,
                            json={"records": prediction_records},
                            timeout=HTTP_TIMEOUT
                        )

                        response.raise_for_status()

                        logger.info(
                            "Windows prediction transmission successful: "
                            "sent=%d, response=%s",
                            len(prediction_records),
                            response.status_code
                        )

                    except requests.RequestException as exc:
                        logger.error(
                            "Windows prediction transmission failed: %s",
                            exc
                        )

                logger.info(
                    "Local CNN batch completed: "
                    "records=%d, preprocessing=%.3f ms, "
                    "inference=%.3f ms, throughput=%.2f records/s",
                    len(fresh_records),
                    preprocessing_time * 1000,
                    inference_time * 1000,
                    len(fresh_records) / inference_time
                    if inference_time > 0 else 0
                )
#.......................................................................................

            except Exception:
                logger.exception(
                    "Local CNN inference failed for batch of %d records",
                    len(batch)
                )

            finally:
                # Mark every retrieved record as processed or discarded.
                for _ in batch:
                    feature_queue.task_done()

    finally:

        # Discard anything still waiting in the queue.
        discarded = 0

        while True:
            try:
                feature_queue.get_nowait()
                feature_queue.task_done()
                discarded += 1

            except queue.Empty:
                break

        if discarded:
            logger.info(
                "Discarded %d pending records during CNN shutdown",
                discarded
            )

        logger.info("Local CNN inference thread stopped")



# ============================================
# MAIN PROCESSING PIPELINE WITH ONOS INTEGRATION
# ============================================
def process_capture_cycle(cycle_num, onos_engine, tracker):
    """Process one capture cycle with deep ONOS integration"""
    logger.info(f"Processing cycle {cycle_num}")
    
    onos_engine.refresh_global_view()
    
    if not capture_flows(5):
        logger.warning("Capture failed")
        return None
    
    process_ndpi()
    
    df = parse_argus_data()
    if df is None or df.empty:
        logger.warning("No flows to process")
        return None
    
    service_map = parse_ndpi_data()
    def get_service(row):
        key = (str(row.get('saddr', '')), str(row.get('sport', '')),
               str(row.get('daddr', '')), str(row.get('dport', '')))
        return service_map.get(key, 'Unknown')
    
    df['service'] = df.apply(get_service, axis=1)
    
    df = calculate_unsw_features(df)
    
    logger.info("Enhancing features with ONOS intelligence...")
    onos_features_list = []
    for _, row in df.iterrows():
        onos_features = onos_engine.correlate_flow_with_onos(row.to_dict())
        onos_features_list.append(onos_features)
    
    onos_df = pd.DataFrame(onos_features_list)
    df = pd.concat([df, onos_df], axis=1)
    
    df = add_connection_tracking_features(df, tracker)
    
    df['timestamp'] = datetime.now().isoformat()
    
    # Step 9: Ensure all features exist and in correct order
    for col in ENHANCED_UNSW_FEATURES:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns
    df = df[ENHANCED_UNSW_FEATURES]
    
  
    logger.info(
        f"Processed {len(df)} flows with ONOS-enhanced feature extraction"
    )
    return df


# ============================================
# MAIN LOOP
# ============================================
def send_onos_features(df):

    if df is None or df.empty:
        return

    feature_columns = [
        "proto",
        "state",
        "dur",
        "sbytes",
        "dbytes",
        "onos_switch_id",
        "onos_flow_priority",
        "onos_path_length"
    ]

    missing = [
        col for col in feature_columns
        if col not in df.columns
    ]

    if missing:
        logger.error("Missing AnDO/ONOS fields: %s", missing)
        return

    records = df[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    ).where(pd.notnull(df[feature_columns]), None).to_dict(
        orient="records"
    )

    for start in range(0, len(records), HTTP_BATCH_SIZE):

        batch = records[start:start + HTTP_BATCH_SIZE]

        try:
            response = requests.post(
                FEATURES_ENDPOINT,
                json={"records": batch},
                timeout=HTTP_TIMEOUT
            )

            response.raise_for_status()

            logger.info(
                "AnDO/ONOS feature transmission successful: sent=%d",
                len(batch)
            )

        except requests.RequestException as exc:
            logger.error(
                "AnDO/ONOS feature transmission failed: %s",
                exc
            )

def main():
    global running
    logger.info("=== ONOS-ENHANCED SDN FEATURE EXTRACTOR ===")
    logger.info(f"ONOS Controller: {ONOS_IP}:{ONOS_PORT}")
    logger.info(f"Monitor Interface: {MON_INTERFACE}")
    logger.info(f"Total Features: {len(ENHANCED_UNSW_FEATURES)}")
    logger.info(f"ONOS-Enhanced Features: 10")
    
    if not check_dependencies():
        return
    
    onos_engine = ONOSIntelligenceEngine(ONOS_IP, ONOS_PORT, ONOS_USER, ONOS_PASS)
    tracker = UNSWConnectionTracker(window_seconds=30)

    # Start IDS batch sender in a background thread
    ids_sender_thread = threading.Thread(
        target=ids_batch_sender,
        name="IDS-Batch-Sender",
        daemon=False
    )

    ids_sender_thread.start()

    logger.info("IDS batch sender thread launched")

       
    cycle = 0
    
    try:
        while running:
            cycle += 1
            df = process_capture_cycle(cycle, onos_engine, tracker)
            
            if df is not None:
                enqueue_model_features(df) # Add the 34 CNN input features to the in-memory queue
                send_onos_features(df)

                if cycle == 1:
                    onos_cols = ['saddr', 'daddr', 'service', 'onos_switch_id', 
                                'onos_flow_priority', 'onos_path_length']
                    print(f"ONOS-Enhanced Sample:\n{df[onos_cols].head(2).to_string()}")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        #logger.info("ONOS-enhanced feature extraction completed")
        logger.info("Stopping local CNN inference worker...")

        # Signal the worker to stop.
        running = False

        # Wait for the worker to finish and exit.
        ids_sender_thread.join()

        logger.info(
            "Remaining CNN queue size: %d",
            feature_queue.qsize()
        )

        logger.info("ONOS-enhanced feature extraction completed")

if __name__ == "__main__":
    main()
