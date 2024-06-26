from sqlalchemy import DateTime, Column, Integer, String, Float

from base_sql import Base

#         closed = candle['c']
#         open = candle['o']
#         high = candle['h']
#         low = candle['l']
#         volume = candle['v']


class CryptoPrice(Base):
    __tablename__ = "crypto"

    id = Column(Integer, primary_key=True)
    crypto_name = Column(String(90))
    close_price = Column(Float())
    open_price = Column(Float())
    high_price = Column(Float())
    low_price = Column(Float())
    volume = Column(Float())
    interval = Column(String(5))
    created_time = Column(DateTime())
    event_time = Column(DateTime())
    open_time = Column(DateTime())
    close_time = Column(DateTime())

    def __int__(self, crypto_name, close_price, open_price, high_price, low_price, volume, interval, created_time, event_time, open_time, close_time):
        self.crypto_name = crypto_name
        self.open_price = open_price
        self.close_price = close_price
        self.high_price = high_price
        self.low_price = low_price
        self.volume = volume
        self.interval = interval
        self.created_time = created_time
        self.event_time = event_time
        self.open_time = open_time
        self.close_time = close_time