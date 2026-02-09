# SPARQL Engine - Mini SPARQL Implementation

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A pedagogical SPARQL engine implemented in Python with a REST Flask API. Designed for learning SPARQL queries and RDF data.

##  Features

### SPARQL Support
- **SELECT** (with `*` or specific variables)
- **WHERE** with triple patterns
- **FILTER** (comparisons: `=`, `!=`, `>`, `<`, `>=`, `<=`)
- **OPTIONAL** (outer joins)
- **UNION** (pattern unions)
- **ORDER BY** (sorting)
- **LIMIT** (result limiting)
- **PREFIX** (prefix definitions)

###  Supported Formats
- **Simplified Turtle (TTL)**
- **Typed literals** (`"123"^^xsd:integer`)
- **IRIs** and prefixed names
- **Numeric data** (integers, floats)

###  REST API
- SPARQL endpoint for query execution
- TTL dataset upload
- Current dataset inspection

##  Installation

### Prerequisites
- Python 3.7+
- pip

### Installation

    git clone https://github.com/Tiho99/sparql-engine.git
    cd sparql-engine
    pip install flask 

## 🎮 Usage

### Start the Server
    python SPRQL.py
