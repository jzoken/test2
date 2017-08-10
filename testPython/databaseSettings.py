# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:59:00 2015

@author: jzoken
"""

import commonSettings


global tireProcessStateDB
tireProcessStateDB = commonSettings.rootDir+"\\tireProcessState.db"

def createStateTransitionTable(c):
    print ("\n creating table")
    cur = c.cursor() 
    cur.execute("CREATE TABLE ts (id integer primary key autoincrement, state integer, path text) ")
    
def addEntry(c,row):
    cur = c.cursor()
    cur.execute("insert into ts (state,path) VALUES(?,?)",row )
    c.commit()

def checkCntTireScansInSpecifiedState(c,tireScanState):
    cur=c.cursor()
    cur.execute("select count(*) from ts where state=?",tireScanState)
    cnt=cur.fetchone()[0]
    print("\n cnt is ",cnt,"\n")
    return(cnt)
    

def getFirstTireScanInSpecifiedState(c,tireScanState):
    cur=c.cursor()
    cur.execute("select * from ts where state=?",tireScanState)
    item = cur.fetchone()
    id=item[0]
    path=item[2]
    print("\n id path1", id, path,"\n")
    return(id,path)


def transitionToNextState(c,id,nextState):
    cur=c.cursor()
    cur.execute("UPDATE ts SET state=? WHERE Id=?", (nextState, id))   
    c.commit()
    

def getFullTireRecord(c,manufacturer,brand,sw,ar,ws):
    cur=c.cursor()
    #cur.execute("select * from deltable2 where Manufacturer=? and Brand=?",("Goodyear","Eagle RS-A"))
    cur.execute("select * from deltable2 where Manufacturer=? and Brand=? and SW=? and AR=? and WS=?",(manufacturer,brand,sw,ar,ws))
    #cur.execute("select * from deltable2 where Manufacturer=?,Brand=?,SW=?,AR=?,WS=?",(manufacturer,brand,sw,ar,ws))
    item = cur.fetchone()
    twimm=item[7]
    print("\n twimm",twimm,"\n")
    return(twimm)