from sqlalchemy import Table, Column, Integer, String, MetaData


class Structured_radiology_report_table:
    def __init__(self, kanjaid, orderno, objectsequence, objectname,
                 objectcertainty):
        self.kanjaid = kanjaid
        self.orderno = orderno
        self.objectsequence = objectsequence
        self.objectname = objectname
        self.objectcertainty = objectcertainty


class Radiology_report:
    def __init__(self, kanjaid, orderno, shoken):
        self.kanjaid = kanjaid
        self.orderno = orderno
        self.shoken = shoken


metadata = MetaData()

structured_table_metadata = Table(
    'structured_radiology_report_table', metadata,
    Column('kanjaid', String, primary_key=True),
    Column('orderno', String, primary_key=True),
    Column('objectsequence', Integer, primary_key=True),
    Column('objectname', String), Column('objectcertainty', String))

radiology_report_metadata = Table('radiology_report_test', metadata,
                                  Column('kanjaid', String, primary_key=True),
                                  Column('orderno', String, primary_key=True),
                                  Column('shoken', String))
