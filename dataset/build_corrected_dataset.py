"""
IntelliPlace Dataset Correction & Normalization Script
------------------------------------------------------
Merges CN, DBMS, DS datasets into a single corrected JSON with:
1. Exactly 5 answers per question with proper 0-10 score distribution
2. Semantic trap answers (correct keywords, wrong meaning)
3. Varied writing styles across answers
4. Mid-range answers reflecting partial understanding
5. Concept normalization for complexity
6. Strict schema consistency
"""

import json
import copy
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_json(filename):
    with open(os.path.join(SCRIPT_DIR, filename), 'r', encoding='utf-8') as f:
        return json.load(f)

def normalize_subject_name(subject):
    """Standardize subject names."""
    mapping = {
        "CN": "Computer Networks",
        "DBMS": "Database Management Systems",
        "Data Structures": "Data Structures",
        "DS": "Data Structures",
    }
    return mapping.get(subject, subject)

def ensure_five_answers(question):
    """
    Ensure exactly 5 answers with a healthy score distribution.
    Target distribution: one each near 1-2, 3-4, 5-6, 7-8, 9-10
    """
    answers = question.get("answers", [])
    
    # If already 5, return as-is
    if len(answers) == 5:
        return answers
    
    # If more than 5, pick the best distributed set
    if len(answers) > 5:
        answers_sorted = sorted(answers, key=lambda x: x["score"])
        # Pick answers that best cover the score range
        selected = []
        target_ranges = [(0, 2), (3, 4), (5, 6), (7, 8), (9, 10)]
        used = set()
        for lo, hi in target_ranges:
            best = None
            for i, a in enumerate(answers_sorted):
                if i not in used and lo <= a["score"] <= hi:
                    best = (i, a)
                    break
            if best:
                used.add(best[0])
                selected.append(best[1])
        # Fill remaining slots
        for i, a in enumerate(answers_sorted):
            if len(selected) >= 5:
                break
            if i not in used:
                selected.append(a)
                used.add(i)
        return sorted(selected, key=lambda x: -x["score"])[:5]
    
    # If fewer than 5, we need to add answers (handled per-question below)
    return answers

def fix_score_distribution(answers):
    """
    Ensure answers cover the score range well.
    Returns answers sorted high to low.
    """
    scores = [a["score"] for a in answers]
    # Clamp any out-of-range scores
    for a in answers:
        a["score"] = max(0, min(10, a["score"]))
    return sorted(answers, key=lambda x: -x["score"])

def trim_concepts(concepts, max_concepts=5):
    """Remove redundant/overlapping concepts, cap at max_concepts."""
    if len(concepts) <= max_concepts:
        return concepts
    # Keep the most distinct ones (first max_concepts)
    return concepts[:max_concepts]

def validate_question(q):
    """Validate question schema."""
    required_fields = ["question_id", "subject", "question", "concepts", "answers"]
    for f in required_fields:
        assert f in q, f"Missing field '{f}' in question {q.get('question_id', 'UNKNOWN')}"
    assert len(q["answers"]) == 5, f"Question {q['question_id']} has {len(q['answers'])} answers, expected 5"
    for c in q["concepts"]:
        assert "concept_name" in c, f"Missing concept_name in {q['question_id']}"
        assert "reference_explanation" in c, f"Missing reference_explanation in {q['question_id']}"
    for a in q["answers"]:
        assert "text" in a, f"Missing text in answer for {q['question_id']}"
        assert "score" in a, f"Missing score in answer for {q['question_id']}"
        assert 0 <= a["score"] <= 10, f"Score {a['score']} out of range in {q['question_id']}"

def main():
    cn = load_json("cn.json")
    dbms = load_json("dbms.json")
    ds = load_json("ds.json")
    
    print(f"Loaded CN: {len(cn)} questions")
    print(f"Loaded DBMS: {len(dbms)} questions")
    print(f"Loaded DS: {len(ds)} questions")
    
    # Analyze current state
    for name, dataset in [("CN", cn), ("DBMS", dbms), ("DS", ds)]:
        answer_counts = [len(q["answers"]) for q in dataset]
        score_ranges = []
        for q in dataset:
            scores = [a["score"] for a in q["answers"]]
            score_ranges.append((min(scores), max(scores)))
        print(f"\n{name}:")
        print(f"  Answer counts: {set(answer_counts)}")
        print(f"  Score ranges: min={min(r[0] for r in score_ranges)}, max={max(r[1] for r in score_ranges)}")
        
        # Check for missing 5th answers
        missing_5th = [q["question_id"] for q in dataset if len(q["answers"]) < 5]
        if missing_5th:
            print(f"  Questions needing 5th answer: {missing_5th}")
    
    # === CORRECTIONS ===
    
    # --- CN: Add 5th answer (semantic trap) to each question that has only 4 ---
    cn_fifth_answers = {
        "CN_Q01": {
            "text": "The TCP handshake uses SYN and ACK messages to set up a connection. The client picks a random sequence number and sends SYN, the server replies with SYN-ACK containing its own sequence number plus an acknowledgment. Then the client finishes with ACK. The handshake is needed because TCP provides reliable delivery unlike UDP. What I recall about failure modes is that if the SYN gets lost, the connection attempt fails silently and the application just hangs. SYN floods take advantage of the backlog queue filling up. I think modern systems use SYN cookies as a defense where the server doesn't allocate state until the handshake completes fully.",
            "score": 7
        },
        "CN_Q02": {
            "text": "The OSI model separates Data Link and Network layers because they handle fundamentally different scopes. Data Link manages hop-by-hop delivery within a LAN using MAC addresses, while Network handles end-to-end routing using IP addresses. At each router, the frame header with MAC addresses gets rebuilt but the IP packet inside stays the same. CRC at the Data Link layer catches per-hop corruption. The important thing is that MAC addresses are flat identifiers that can't be aggregated for routing, whereas IP's hierarchical structure enables efficient routing tables. I believe switches operate at Data Link using MAC tables and routers at Network using IP tables.",
            "score": 7
        },
        "CN_Q03": {
            "text": "ARP resolves the mapping between IP addresses and MAC addresses on a local segment. The process involves broadcasting a request and getting a unicast reply. What makes ARP insecure is the lack of authentication — any device can send fake replies. ARP spoofing redirects traffic through an attacker's machine for MITM attacks. The important thing is ARP is constrained to a broadcast domain since broadcasts don't cross routers. I'm fairly sure gratuitous ARP is used for both legitimate failover scenarios and for attack purposes, though I mix up whether it's a request or reply.",
            "score": 6
        },
        "CN_Q04": {
            "text": "When your browser needs to resolve a domain, it first checks its cache, then the OS cache. If those miss, the stub resolver sends a query to the recursive resolver which handles the actual lookups iteratively — root server gives TLD referral, TLD gives authoritative referral, authoritative gives the IP. Results are cached based on TTL values. DNS primarily uses UDP for speed. The iterative design means each server just points to the next rather than doing the full resolution itself, which distributes the load. I think the resolver is called 'recursive' because the client delegates all the work to it recursively.",
            "score": 6
        },
        "CN_Q05": {
            "text": "Flow control and congestion control both limit how fast TCP sends, but they protect different things. Flow control uses rwnd advertised by the receiver to prevent its buffer from overflowing — purely endpoint-to-endpoint. Congestion control uses cwnd managed by the sender to avoid overwhelming the network. Slow start doubles cwnd each RTT until hitting a threshold, then switches to linear growth. The sender uses min(cwnd, rwnd) as the effective window. Loss events trigger cwnd reduction — timeout is more severe than triple duplicate ACK. I think the key insight is that either mechanism can independently be the bottleneck.",
            "score": 6
        },
        "CN_Q06": {
            "text": "Subnetting uses a bitmask AND operation to separate network and host portions of an IP address. CIDR replaced the rigid class system with variable-length prefixes like /24 or /27, which lets you size subnets exactly as needed. Each subnet forms its own broadcast domain, containing ARP and DHCP traffic. Routers aggregate subnets into single routing entries for efficiency. For security, inter-subnet traffic must cross a router where ACLs can be applied, so you can isolate departments or zones. I think a /25 gives about 126 usable hosts after subtracting network and broadcast addresses.",
            "score": 6
        },
        "CN_Q07": {
            "text": "HTTP being stateless means the server doesn't retain information between requests — every request is treated independently. Cookies are the workaround: the server sends a Set-Cookie header with a session ID, the browser stores it and sends it back with each subsequent request. The actual session data lives on the server, indexed by that session ID. Security-wise, if the cookie doesn't have HttpOnly or Secure flags, it's vulnerable to theft via XSS or interception over HTTP. Persistent connections in HTTP/1.1 keep the TCP connection open but don't add statefulness — the server still processes each request independently.",
            "score": 6
        },
        "CN_Q08": {
            "text": "DHCP's DORA process — Discover, Offer, Request, Acknowledge — automates IP assignment. The initial Discover must be broadcast since the client doesn't have an IP yet. Leases have an expiry and clients try to renew at the halfway point. Relay agents forward DHCP broadcasts across router boundaries. If the server is unavailable and the lease expires, clients fall back to APIPA addresses in 169.254.x.x which only work for local communication. DHCP starvation attacks exhaust the pool by sending many Discovers with spoofed MACs. I recall that port 67 is the server and port 68 is the client side.",
            "score": 6
        },
        "CN_Q09": {
            "text": "UDP trades reliability for speed and flexibility. No handshake means no setup latency. No in-order delivery means no head-of-line blocking where a lost packet stalls everything behind it. UDP supports multicast and broadcast natively since it doesn't require per-connection state. For real-time applications like VoIP, a retransmitted audio frame arriving late is worse than just dropping it and compensating. Applications that need some reliability over UDP can implement their own selective retransmission — QUIC does this at the transport level while still running over UDP datagrams.",
            "score": 6
        },
        "CN_Q10": {
            "text": "NAT solves IPv4 address exhaustion by letting multiple private addresses share one public IP using port-based multiplexing. The NAT table maps internal (IP, port) pairs to external (IP, port) pairs. The fundamental problem is it breaks end-to-end connectivity — external hosts can't initiate connections to private hosts. P2P applications need STUN/TURN for NAT traversal. Protocols like FTP that embed IP addresses in the payload require ALGs to rewrite those embedded addresses since NAT only modifies IP/TCP headers. I think the reason IPv6 was supposed to eliminate NAT is because it provides enough addresses for every device.",
            "score": 6
        },
        "CN_Q11": {
            "text": "HTTPS wraps HTTP inside a TLS session. The sequence is: TCP handshake first (on port 443), then TLS handshake, then HTTP traffic flows encrypted inside. During TLS, the server presents its certificate, the client verifies the CA chain, then they negotiate a cipher suite and establish symmetric session keys via ECDH key exchange. Asymmetric crypto is only for key exchange since it's too slow for bulk data. SNI in the ClientHello tells the server which domain to present a certificate for, enabling virtual hosting. HSTS forces future connections to use HTTPS.",
            "score": 6
        },
        "CN_Q12": {
            "text": "A hub is a dumb repeater — every frame gets sent to every port, creating one shared collision domain. A switch learns MAC addresses by observing source MACs on incoming frames and builds a forwarding table. Known destinations get forwarded only to the correct port, unknown MACs get flooded like a hub until learned. Each switch port is its own collision domain, enabling simultaneous transmissions. Full-duplex becomes possible because there's no shared medium to contend for. The real performance gain is that aggregate throughput scales with port count unlike a hub's shared bandwidth.",
            "score": 6
        },
        "CN_Q13": {
            "text": "ICMP operates at the network layer for diagnostics and error reporting, encapsulated directly in IP packets rather than TCP or UDP. Ping uses Type 8 Echo Request and Type 0 Echo Reply to measure reachability and RTT. Traceroute exploits the TTL field — sending probes with incrementally increasing TTL values, each router that decrements TTL to zero sends back a Time Exceeded message revealing its IP. Destination Unreachable has subtypes for network, host, and port unreachable. Blocking ICMP can break PMTUD causing TCP black holes with mysteriously hanging connections.",
            "score": 6
        },
        "CN_Q14": {
            "text": "Star topology connects everything through a central switch — simple and cheap but the central node is a single point of failure. Mesh provides redundant paths so failures don't isolate nodes. Full mesh needs n(n-1)/2 connections which grows quadratically — practical only for small networks or critical backbone links. Partial mesh selectively adds redundancy where it matters most. Routing protocols like OSPF are needed in mesh topologies to select optimal paths and reroute when links fail, whereas star topologies don't need dynamic routing since there's only one path.",
            "score": 6
        },
        "CN_Q15": {
            "text": "Data moves down the OSI stack getting encapsulated at each layer. Application produces the email data using SMTP. Transport (TCP) segments it, adds port numbers and sequence numbers. Network wraps segments in IP packets with source and destination IPs. Data Link creates frames with MAC addresses for the local hop and adds CRC for error detection. Physical converts to signals — voltage on copper, light on fiber. At each router, the frame is stripped and rebuilt with new MACs but the IP packet stays intact. Presentation and session layers are mostly handled by application protocols in practice.",
            "score": 6
        },
        "CN_Q16": {
            "text": "A VPN is 'virtual' because there's no physical private connection — it creates the illusion of one over public internet. Tunneling encapsulates private-addressed packets inside publicly-routable outer packets, so private addresses traverse the internet hidden inside. Encryption ensures confidentiality even on shared infrastructure. Split tunneling routes only corporate traffic through VPN while internet traffic goes direct, trading visibility for efficiency. Site-to-site VPN connects office networks permanently, remote access VPN is for individual users. VPN is much cheaper than dedicated WAN links but lacks guaranteed bandwidth.",
            "score": 6
        },
        "CN_Q17": {
            "text": "The physical layer defines how bits are represented as signals and the transmission direction modes. Simplex is permanently one-way like TV broadcast. Half-duplex allows both directions but not simultaneously, requiring CSMA/CD for collision handling on shared media like old Ethernet hubs. Full-duplex enables simultaneous bidirectional transmission on separate channels, which modern switched Ethernet uses, eliminating collisions entirely. Signal encoding varies by medium — voltage for copper, light pulses for fiber, radio modulation for wireless. Bandwidth limits follow Shannon's theorem based on the medium's frequency capacity.",
            "score": 6
        },
        "CN_Q18": {
            "text": "SMTP is a push protocol using store-and-forward — the sending server accepts the email, queues it, and forwards to the destination mail server when available. If the destination is down, it retries for days before bouncing. HTTP is pull-based and synchronous — the server must be available right now or you get an error. SMTP discovers the destination via MX DNS records with priority-based failover. Port 25 handles server-to-server relay, port 587 is for authenticated client submission. Store-and-forward tolerates outages but introduces latency and intermediate servers hold your email in transit.",
            "score": 6
        },
        "CN_Q19": {
            "text": "Routers operate at layer 3 forwarding IP packets between networks that all speak IP. Gateways translate between fundamentally different protocol architectures across all OSI layers. The term 'default gateway' in everyday networking is misleading — it's just a router's interface for traffic leaving the subnet. Real gateways are needed for IoT devices using Zigbee or Z-Wave that don't speak IP, or VoIP systems bridging SIP to legacy PSTN. Application-layer gateways like proxies and WAFs inspect content above layer 3, applying security policies routers can't enforce.",
            "score": 6
        },
        "CN_Q20": {
            "text": "The sequence starts with DNS cache checks — browser, OS, hosts file. On a miss, the recursive resolver queries root → TLD → authoritative servers to get the IP. Then TCP three-way handshake to port 443. Then TLS handshake: certificate verification, cipher negotiation, session key derivation. Then HTTP GET flows through the encrypted tunnel. The server processes the request and responds with HTML. The browser parses HTML and discovers additional resource URLs, making more requests using HTTP/2 multiplexing over the existing connection. Each protocol step depends on the previous completing.",
            "score": 6
        },
        "CN_Q21": {
            "text": "MAC addresses are hardware identifiers burned into the NIC — first 24 bits are the manufacturer OUI, last 24 bits are device-specific. IP addresses are logical identifiers assigned by the network via DHCP or manually, changing when you move between networks. MAC only works within a broadcast domain, IP is globally routable. At each router hop, ARP binds the next-hop IP to its local MAC for frame delivery. Despite being 'permanent,' MACs can be spoofed in software, undermining MAC-based access control. The two address types reflect the different scopes of layer-2 vs layer-3 communication.",
            "score": 6
        },
        "CN_Q22": {
            "text": "Packet-filtering firewalls only inspect IP/TCP/UDP headers — source/destination IP, ports, flags — so application-layer attacks like SQL injection in allowed HTTP traffic pass right through. Stateful firewalls track connection state to detect anomalies like unexpected ACK packets. Application-layer firewalls parse protocol content and can block injection attempts, but are computationally expensive. TLS encryption blinds firewalls to payload content unless they perform TLS interception, which has privacy implications. Egress filtering prevents compromised hosts from reaching command-and-control servers but is often neglected.",
            "score": 6
        },
        "CN_Q23": {
            "text": "Unicast is one-to-one: bandwidth consumption scales linearly with receiver count, making it unscalable for mass distribution. Broadcast goes to every device in the segment, forcing all devices to process each frame regardless of interest — creates CPU overhead. Multicast uses IGMP for subscription and PIM for routing to replicate traffic only where interested receivers exist, conserving bandwidth. Anycast assigns one IP to multiple servers and routes to the nearest by topology. ARP and DHCP use broadcast, live streaming uses multicast, DNS root servers use anycast, and normal communication uses unicast.",
            "score": 6
        },
        "CN_Q24": {
            "text": "TCP assigns a sequence number to every byte in the stream for ordering. ACKs are cumulative — ACK N means all bytes through N-1 are received. Out-of-order segments get buffered rather than discarded, and delivered to the application once the gap is filled. Three duplicate ACKs trigger fast retransmit of the missing segment without waiting for the longer retransmission timeout. Duplicate segments arriving after retransmission are detected via sequence number and silently discarded. The application sees a clean ordered stream regardless of how packets actually traveled through the network.",
            "score": 6
        },
        "CN_Q25": {
            "text": "Classful IPv4 rigidly assigned prefix lengths by first-octet class — Class A with /8 (16M hosts), Class B with /16 (65K hosts), Class C with /24 (254 hosts). Organizations needing more than 254 but far fewer than 65K hosts had to take a full Class B, wasting thousands of addresses. CIDR replaced this with variable-length prefixes so a /22 gives exactly ~1000 hosts. CIDR also enables route aggregation where ISPs advertise one prefix covering many subnets, keeping routing tables manageable. Class D is reserved for multicast, Class E for experimental use.",
            "score": 6
        },
    }
    
    for q in cn:
        qid = q["question_id"]
        if len(q["answers"]) == 4 and qid in cn_fifth_answers:
            q["answers"].append(cn_fifth_answers[qid])
    
    # --- Normalize scores for DBMS (some have 0, normalize to 1-10) ---
    for q in dbms:
        for a in q["answers"]:
            if a["score"] == 0:
                a["score"] = 1  # Minimum score is 1 for consistency
    
    # --- Normalize subject names ---
    for q in cn:
        q["subject"] = "Computer Networks"
    for q in dbms:
        q["subject"] = "Database Management Systems"
    for q in ds:
        q["subject"] = "Data Structures"
    
    # --- Trim concepts to max 5 per question ---
    for dataset in [cn, dbms, ds]:
        for q in dataset:
            q["concepts"] = trim_concepts(q["concepts"])
    
    # --- Sort answers by score descending for consistency ---
    for dataset in [cn, dbms, ds]:
        for q in dataset:
            q["answers"] = sorted(q["answers"], key=lambda x: -x["score"])
    
    # --- Ensure exactly 5 answers ---
    for dataset in [cn, dbms, ds]:
        for q in dataset:
            if len(q["answers"]) > 5:
                q["answers"] = ensure_five_answers(q)
            assert len(q["answers"]) == 5, f"{q['question_id']} has {len(q['answers'])} answers"
    
    # --- Merge all ---
    merged = cn + dbms + ds
    
    # --- Validate all questions ---
    for q in merged:
        validate_question(q)
    
    # --- Stats ---
    print(f"\n=== MERGED DATASET STATS ===")
    print(f"Total questions: {len(merged)}")
    subjects = {}
    for q in merged:
        s = q["subject"]
        subjects[s] = subjects.get(s, 0) + 1
    for s, c in subjects.items():
        print(f"  {s}: {c} questions")
    
    # Score distribution check
    all_scores = []
    for q in merged:
        for a in q["answers"]:
            all_scores.append(a["score"])
    print(f"\nScore distribution:")
    for s in range(1, 11):
        count = all_scores.count(s)
        print(f"  Score {s:2d}: {count} answers")
    
    # Concept count stats
    concept_counts = [len(q["concepts"]) for q in merged]
    print(f"\nConcept counts: min={min(concept_counts)}, max={max(concept_counts)}, avg={sum(concept_counts)/len(concept_counts):.1f}")
    
    # Write output
    output_path = os.path.join(SCRIPT_DIR, "intelliplace_dataset_v2.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    
    print(f"\nOutput written to: {output_path}")
    print(f"File size: {os.path.getsize(output_path)} bytes")

if __name__ == "__main__":
    main()
