#!/bin/bash

SAVEPATH="../bin/data/saves/qsave.sav"
sqlite3 $SAVEPATH "delete from planets;"
g++ setgen.cpp
./a.out | sqlite3 $SAVEPATH
