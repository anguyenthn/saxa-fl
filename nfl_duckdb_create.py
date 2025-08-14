#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 21:36:13 2025

@author: an
"""

import os, duckdb
os.chdir('/Users/an/Downloads/Projects/NFL/Data')

conn = duckdb.connect('nfl.duckdb')
conn.execute("CREATE OR REPLACE TABLE nfl AS SELECT * FROM read_csv_auto('NFLpbp_Schedule_Merged.csv');")
conn.close()
