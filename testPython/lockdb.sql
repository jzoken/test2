.open testlock.db
--CREATE TABLE lt (id integer primary key, markerID integer, state integer, path text);
CREATE TABLE lt (id integer primary key autoincrement, markerID integer, state integer, path text);
CREATE TABLE markerTable (id integer primary key, path text);
