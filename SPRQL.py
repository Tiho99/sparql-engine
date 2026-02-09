#!/usr/bin/env python3
"""
sparql_mini.py

Petit moteur SPARQL pédagogique (sous-ensemble) + API Flask.

Install:
    pip install flask

Usage:
    python sparql_mini.py

Endpoints:
    POST /sparql        -> body text/plain with SPARQL query or JSON {"query": "..."}
    POST /upload        -> upload TTL-like text (JSON {"ttl": "..."} or raw text)
    GET  /dataset       -> returns current dataset (triples)

Notes:
    - This is NOT a full SPARQL engine. It's an educational subset useful for TP.
    - Supports: PREFIX, SELECT (?vars or *), WHERE { ... }, FILTER (simple),
                OPTIONAL { ... }, UNION { ... }, ORDER BY ?var, LIMIT n
    - TTL format accepted (very simple): PREFIX ex: <http://...#>
      ex:Alice ex:age "30" .
"""

from flask import Flask, request, jsonify
import re
from typing import List, Tuple, Dict, Any, Union

# ---------------- Types ----------------
Triple = Tuple[str, str, Any]  # subject, predicate, object (object can be str/int/float)
Dataset = List[Triple]

# ---------------- Utilities ----------------
def normalize_literal(token: str) -> Union[int, float, str]:
    token = token.strip()
    # typed literal "123"^^xsd:integer
    m = re.match(r'^"(.+)"\^\^<?([^>]+)>?$', token)
    if m:
        val, dtype = m.group(1), m.group(2).lower()
        if 'int' in dtype or 'integer' in dtype:
            try:
                return int(val)
            except:
                return val
        if 'float' in dtype or 'double' in dtype or 'decimal' in dtype:
            try:
                return float(val)
            except:
                return val
        return val
    # plain quoted literal
    if token.startswith('"') and token.endswith('"'):
        inner = token[1:-1]
        # try int
        if re.fullmatch(r'-?\d+', inner):
            return int(inner)
        if re.fullmatch(r'-?\d+\.\d+', inner):
            return float(inner)
        return inner
    # plain numeric
    if re.fullmatch(r'-?\d+\.\d+', token):
        return float(token)
    if re.fullmatch(r'-?\d+', token):
        return int(token)
    return token  # likely a prefixed name or IRI

def expand_prefixed(token: str, prefixes: Dict[str,str]) -> str:
    token = token.strip()
    if token.startswith('<') and token.endswith('>'):
        return token[1:-1]
    if token.startswith('"') and token.endswith('"'):
        return token  # leave literal form for now
    if ':' in token:
        pref, local = token.split(':', 1)
        if pref in prefixes:
            return prefixes[pref] + local
    return token

# ---------------- TTL loader ----------------
def load_simple_ttl_text(text: str) -> Tuple[Dict[str,str], Dataset]:
    prefixes: Dict[str,str] = {}
    triples: Dataset = []
    lines = text.splitlines()
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith('#'):
            continue
        # PREFIX lines like: PREFIX ex: <http://example.com#>
        m = re.match(r'PREFIX\s+(\w+):\s*<([^>]+)>', ln, re.I)
        if m:
            prefixes[m.group(1)] = m.group(2)
            continue
        # triple line: subj pred obj .
        # naive parse: handle quoted object with spaces
        # remove terminal dot
        if ln.endswith('.'):
            ln = ln[:-1].strip()
        parts = re.split(r'\s+', ln, maxsplit=2)
        if len(parts) < 3:
            continue
        s_raw, p_raw, o_raw = parts[0], parts[1], parts[2]
        # handle quoted literal containing spaces (if o_raw starts with " but not ends)
        if o_raw.startswith('"') and not o_raw.endswith('"'):
            # try to find quoted substring in original line
            qm = re.search(r'(".*")', ln)
            if qm:
                o_raw = qm.group(1)
        s = expand_prefixed(s_raw, prefixes)
        p = expand_prefixed(p_raw, prefixes)
        # object expand if IRI/prefixed, else normalize literal
        if o_raw.startswith('<') or (':' in o_raw and not o_raw.startswith('"')):
            o_expanded = expand_prefixed(o_raw, prefixes)
            # if it remains with quotes it's a literal, handle below
            if isinstance(o_expanded, str) and o_expanded.startswith('"') and o_expanded.endswith('"'):
                o_val = normalize_literal(o_expanded)
            else:
                o_val = o_expanded
        else:
            o_val = normalize_literal(o_raw)
        triples.append((s, p, o_val))
    return prefixes, triples

# ---------------- Query parser ----------------
def extract_braced_blocks(s: str) -> List[str]:
    """Extract top-level {...} blocks (non-nested expected)."""
    blocks = []
    stack = []
    start = None
    for i, ch in enumerate(s):
        if ch == '{':
            if start is None:
                start = i + 1
            stack.append('{')
        elif ch == '}':
            if stack:
                stack.pop()
                if not stack and start is not None:
                    blocks.append(s[start:i])
                    start = None
    return blocks

def parse_query(query: str) -> Dict[str, Any]:
    # normalize whitespace
    q = query.strip()
    lines = [ln.strip() for ln in q.splitlines() if ln.strip()]
    prefixes: Dict[str,str] = {}
    select_vars: List[str] = []
    where_text = ""
    order_by = None
    limit = None

    # Extract PREFIX and SELECT and ORDER BY / LIMIT
    whole = " ".join(lines)
    # PREFIX
    for m in re.finditer(r'PREFIX\s+(\w+):\s*<([^>]+)>', whole, re.I):
        prefixes[m.group(1)] = m.group(2)
    # SELECT
    m_sel = re.search(r'SELECT\s+(DISTINCT\s+)?(.+?)\s+WHERE', whole, re.I | re.S)
    if m_sel:
        sel_body = m_sel.group(2).strip()
        if sel_body == '*':
            select_vars = ['*']
        else:
            select_vars = re.findall(r'\?\w+', sel_body)
    else:
        # fallback: SELECT at start no WHERE (rare)
        m_sel2 = re.search(r'SELECT\s+(.+)', whole, re.I)
        if m_sel2:
            sel_body = m_sel2.group(1)
            select_vars = re.findall(r'\?\w+', sel_body)
            if not select_vars:
                select_vars = ['*']

    # ORDER BY
    m_ob = re.search(r'ORDER\s+BY\s+(\?\w+)', whole, re.I)
    if m_ob:
        order_by = m_ob.group(1)
    # LIMIT
    m_lim = re.search(r'LIMIT\s+(\d+)', whole, re.I)
    if m_lim:
        limit = int(m_lim.group(1))

    # WHERE: capture the first top-level {...}
    where_blocks = extract_braced_blocks(whole)
    if where_blocks:
        # Support UNION by splitting at " } UNION { " patterns — we've extracted blocks individually so
        where_text = " UNION ".join([blk.strip() for blk in where_blocks])
    else:
        # fallback, try to get content between first { and last }
        m_w = re.search(r'WHERE\s*\{(.*)\}', whole, re.I | re.S)
        where_text = m_w.group(1).strip() if m_w else ""

    # Now parse where_text into groups separated by 'UNION'
    raw_groups = [g.strip() for g in re.split(r'\bUNION\b', where_text, flags=re.I) if g.strip()]

    parsed_groups = []
    for g in raw_groups:
        # find OPTIONAL blocks and remove them
        optionals = re.findall(r'OPTIONAL\s*\{([^}]*)\}', g, re.I | re.S)
        main = re.sub(r'OPTIONAL\s*\{[^}]*\}', '', g, flags=re.I | re.S)
        # split main into statements by '.' (triples or FILTERs)
        stmts = [st.strip() for st in re.split(r'\s*\.\s*', main) if st.strip()]
        triples = []
        filters = []
        for st in stmts:
            if not st:
                continue
            if st.upper().startswith('FILTER'):
                fm = re.search(r'FILTER\s*\((.*)\)', st, re.I | re.S)
                if fm:
                    filters.append(fm.group(1).strip())
                continue
            parts = re.split(r'\s+', st, maxsplit=2)
            if len(parts) >= 3:
                s,p,o = parts[0].strip(), parts[1].strip(), parts[2].strip()
                triples.append((s,p,o))
        # parse optional triple groups
        optional_triples = []
        for op in optionals:
            opt_stmts = [st.strip() for st in re.split(r'\s*\.\s*', op) if st.strip()]
            opt_ts = []
            for ost in opt_stmts:
                if not ost:
                    continue
                if ost.upper().startswith('FILTER'):
                    # ignore filter inside optional for now
                    continue
                parts = re.split(r'\s+', ost, maxsplit=2)
                if len(parts) >= 3:
                    opt_ts.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
            if opt_ts:
                optional_triples.append(opt_ts)
        parsed_groups.append({'triples': triples, 'optionals': optional_triples, 'filters': filters})
    return {
        'prefixes': prefixes,
        'select': select_vars,
        'groups': parsed_groups,
        'order_by': order_by,
        'limit': limit
    }

# ---------------- Matching & Execution ----------------
def is_var(tok: str) -> bool:
    return isinstance(tok, str) and tok.startswith('?')

def expand_pattern_term(term: str, prefixes: Dict[str,str]) -> Any:
    # If it's a literal like "30" return normalized literal; if prefixed/IRI, expand to full IRI string
    term = term.strip()
    if term.startswith('"') and term.endswith('"'):
        return normalize_literal(term)
    if term.startswith('<') and term.endswith('>'):
        return term[1:-1]
    # if contains ':' and a known prefix:
    if ':' in term:
        pref = term.split(':',1)[0]
        if pref in prefixes:
            return prefixes[pref] + term.split(':',1)[1]
    # if it's numeric or bare literal
    norm = normalize_literal(term)
    return norm

def unify_pattern_with_triple(pattern, triple: Triple, prefixes: Dict[str,str], binding: Dict[str,Any]) -> Union[Dict[str,Any], None]:
    s_p, p_p, o_p = pattern
    s_t, p_t, o_t = triple
    new = dict(binding)
    for qterm, dterm in ((s_p, s_t), (p_p, p_t), (o_p, o_t)):
        if is_var(qterm):
            if qterm in new:
                # must be same
                if new[qterm] != dterm:
                    return None
            else:
                new[qterm] = dterm
        else:
            expected = expand_pattern_term(qterm, prefixes)
            # If expected is a string representing an IRI, compare directly.
            if isinstance(expected, str) and isinstance(dterm, str):
                if expected != dterm:
                    return None
            else:
                # compare primitive values
                if expected != dterm:
                    return None
    return new

def apply_filters(bindings: List[Dict[str,Any]], filters: List[str]) -> List[Dict[str,Any]]:
    if not filters:
        return bindings
    out = []
    for b in bindings:
        ok_all = True
        for f in filters:
            # simple single comparison: ?var op value
            m = re.match(r'\s*(\?\w+)\s*(>=|<=|!=|=|>|<)\s*(.+)\s*', f)
            if not m:
                ok_all = False
                break
            var, op, val_raw = m.group(1), m.group(2), m.group(3).strip()
            if var not in b:
                ok_all = False
                break
            left = b[var]
            # normalize right value
            if val_raw.startswith('"') and val_raw.endswith('"'):
                right = val_raw[1:-1]
            elif re.fullmatch(r'-?\d+\.\d+', val_raw):
                right = float(val_raw)
            elif re.fullmatch(r'-?\d+', val_raw):
                right = int(val_raw)
            else:
                # treat as IRI or prefixed - in this simple engine, compare as raw string
                right = val_raw
            try:
                if op == '=':
                    ok = (left == right)
                elif op == '!=':
                    ok = (left != right)
                elif op == '>':
                    ok = (left > right)
                elif op == '<':
                    ok = (left < right)
                elif op == '>=':
                    ok = (left >= right)
                elif op == '<=':
                    ok = (left <= right)
                else:
                    ok = False
            except Exception:
                ok = False
            if not ok:
                ok_all = False
                break
        if ok_all:
            out.append(b)
    return out

def join_patterns(triples_patterns: List[Tuple[str,str,str]], data: Dataset, prefixes: Dict[str,str]) -> List[Dict[str,Any]]:
    # naive backtracking join across pattern list
    bindings = [ {} ]  # list of dicts
    for pat in triples_patterns:
        new_bindings = []
        for b in bindings:
            for t in data:
                ub = unify_pattern_with_triple(pat, t, prefixes, b)
                if ub is not None:
                    new_bindings.append(ub)
        bindings = new_bindings
        if not bindings:
            break
    return bindings

def apply_optional_bindings(bindings: List[Dict[str,Any]], optional_patterns_list: List[List[Tuple[str,str,str]]], data: Dataset, prefixes: Dict[str,str]) -> List[Dict[str,Any]]:
    # optional_patterns_list: list of optional groups (each is list of triple patterns)
    # For each binding, try to extend with optional matches; if none, keep original binding
    result = []
    for b in bindings:
        current = [b]
        for opt_group in optional_patterns_list:
            extended = []
            for cb in current:
                # attempt to match all patterns in opt_group with cb as initial binding
                matches = [cb]
                for pat in opt_group:
                    new_matches = []
                    for m in matches:
                        for t in data:
                            ub = unify_pattern_with_triple(pat, t, prefixes, m)
                            if ub is not None:
                                new_matches.append(ub)
                    matches = new_matches
                if matches:
                    extended.extend(matches)
                else:
                    extended.append(cb)
            current = extended
        result.extend(current)
    return result

def execute_parsed_query(parsed: Dict[str,Any], data: Dataset) -> List[Dict[str,Any]]:
    prefixes = parsed['prefixes']
    groups = parsed['groups']
    all_bindings = []
    for group in groups:
        triples = group['triples']
        optionals = group['optionals']
        filters = group['filters']
        # join main triples
        bindings = join_patterns(triples, data, prefixes) if triples else [{}]
        # apply filters
        bindings = apply_filters(bindings, filters)
        # apply optionals
        if optionals:
            bindings = apply_optional_bindings(bindings, optionals, data, prefixes)
        all_bindings.extend(bindings)
    return all_bindings

def project_bindings(bindings: List[Dict[str,Any]], select_vars: List[str]) -> List[Dict[str,Any]]:
    projected = []
    for b in bindings:
        if select_vars == ['*']:
            # return full binding with human-friendly values
            projected.append({k: format_value(v) for k,v in b.items()})
        else:
            row = {}
            for v in select_vars:
                row[v] = format_value(b.get(v))
            projected.append(row)
    return projected

def format_value(v):
    if v is None:
        return None
    # primitive (int/float) or string. If string is an IRI (starts with http or contains ':'?), return as-is
    return v

# ---------------- High-level API ----------------
def execute_query(query: str, data: Dataset) -> Dict[str,Any]:
    parsed = parse_query(query)
    # Merge prefixes from data loader? Data loader prefixes used only when loading the dataset; parsed prefixes used for interpreting patterns
    bindings = execute_parsed_query(parsed, data)
    # projection
    projected = project_bindings(bindings, parsed['select'])
    # order by
    order_by = parsed.get('order_by')
    if order_by:
        # simple sort: None at end
        projected.sort(key=lambda r: (r.get(order_by) is None, r.get(order_by)))
    # limit
    limit = parsed.get('limit')
    if limit is not None:
        projected = projected[:limit]
    return {'results': projected, 'count': len(projected)}

# ---------------- In-memory dataset (default) ----------------
DEFAULT_TTL = """PREFIX ex: <http://example.com#>
ex:Alice ex:age "30" .
ex:Bob ex:age "25" .
ex:Alice ex:knows ex:Bob .
ex:Bob ex:knows ex:Charlie .
ex:Charlie ex:age "35" .
"""

GLOBAL_PREFIXES, GLOBAL_DATA = load_simple_ttl_text(DEFAULT_TTL)

# ---------------- Flask app ----------------
app = Flask(__name__)

@app.route('/dataset', methods=['GET'])
def get_dataset():
    disp = []
    for s,p,o in GLOBAL_DATA:
        disp.append({'s': s, 'p': p, 'o': o})
    return jsonify({'triples': disp, 'prefixes': GLOBAL_PREFIXES})

@app.route('/upload', methods=['POST'])
def upload_ttl():
    # Accept JSON {"ttl": "..."} or raw text body
    txt = None
    if request.is_json:
        txt = request.json.get('ttl')
    else:
        txt = request.get_data(as_text=True)
    if not txt:
        return jsonify({'error': 'No TTL provided'}), 400
    prefixes, triples = load_simple_ttl_text(txt)
    # replace global dataset
    global GLOBAL_PREFIXES, GLOBAL_DATA
    GLOBAL_PREFIXES = prefixes
    GLOBAL_DATA = triples
    return jsonify({'message': 'Dataset loaded', 'prefixes': prefixes, 'triples_count': len(triples)})

@app.route('/sparql', methods=['POST'])
def sparql_endpoint():
    q = None
    if request.is_json:
        q = request.json.get('query')
    else:
        q = request.get_data(as_text=True)
    if not q:
        return jsonify({'error': 'No query provided'}), 400
    res = execute_query(q, GLOBAL_DATA)
    return jsonify(res)

# ---------------- CLI / Demo ----------------
def demo():
    print("Demo dataset (triples):")
    for t in GLOBAL_DATA:
        print(" ", t)
    q1 = """
    PREFIX ex: <http://example.com#>
    SELECT ?person ?age WHERE {
      ?person ex:age ?age .
      FILTER (?age > 28)
    }
    """
    print("\nQuery 1:\n", q1.strip())
    out1 = execute_query(q1, GLOBAL_DATA)
    print("Result:", out1)
    q2 = """
    PREFIX ex: <http://example.com#>
    SELECT ?s ?o WHERE {
      { ?s ex:knows ?o . } UNION { ?s ex:age ?o . }
    } LIMIT 10
    """
    print("\nQuery 2:\n", q2.strip())
    out2 = execute_query(q2, GLOBAL_DATA)
    print("Result:", out2)
    q3 = """
    PREFIX ex: <http://example.com#>
    SELECT ?s ?age WHERE {
      ?s ex:age ?age .
      OPTIONAL { ?s ex:knows ?friend . }
    } ORDER BY ?age LIMIT 10
    """
    print("\nQuery 3:\n", q3.strip())
    out3 = execute_query(q3, GLOBAL_DATA)
    print("Result:", out3)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="sparql_mini - educational SPARQL subset engine with Flask")
    parser.add_argument('--demo', action='store_true', help='Run demo and exit')
    parser.add_argument('--host', default='0.0.0.0', help='Flask host')
    parser.add_argument('--port', default=5000, type=int, help='Flask port')
    args = parser.parse_args()
    if args.demo:
        demo()
    else:
        print("Starting sparql_mini Flask app on http://%s:%d" % ('0.0.0.0', args.port))
        print("GET  /dataset  -> view dataset")
        print("POST /upload   -> upload TTL text (body raw or JSON {\"ttl\":\"...\"})")
        print("POST /sparql   -> run SPARQL query (body raw or JSON {\"query\":\"...\"})")
        app.run(host=args.host, port=args.port, debug=True)
