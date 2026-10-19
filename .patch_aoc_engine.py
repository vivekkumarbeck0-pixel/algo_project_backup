from pathlib import Path

path = Path(r'c:\Users\asus\OneDrive\Documents\algo_project\engine\aoc_engine.py')
path.write_text('''class AOCEngine:

    LEVEL_TOLERANCE = 5

    def __init__(self):

        self.green_strike = None
        self.red_strike = None

        self.green_support = None
        self.red_resistance = None

        self.current_price = None
        self.scenario = None

        self.touch_count = {}

    def load_data(self, aoc_data):
        self.current_price = (
            aoc_data.get("current_price")
            or aoc_data.get("spot")
            or aoc_data.get("market_strike")
        )

        self.green_strike = aoc_data.get("green_strike")
        self.red_strike = aoc_data.get("red_strike")

        self.green_support = (
            aoc_data.get("green_support")
            or aoc_data.get("support_strike")
        )
        self.red_resistance = (
            aoc_data.get("red_resistance")
            or aoc_data.get("resistance_strike")
        )

    def detect_scenario(self):
        if self.current_price is None:
            self.scenario = None
            return

        support = self.green_support
        resistance = self.red_resistance
        price = self.current_price
        tol = self.LEVEL_TOLERANCE

        if support is not None and abs(price - support) <= tol:
            self.scenario = "near_support"
        elif resistance is not None and abs(price - resistance) <= tol:
            self.scenario = "near_resistance"
        elif support is not None and resistance is not None:
            if price < support:
                self.scenario = "below_support"
            elif price > resistance:
                self.scenario = "above_resistance"
            else:
                self.scenario = "between_support_resistance"
        elif self.green_strike is not None and price <= self.green_strike:
            self.scenario = "below_green"
        elif self.red_strike is not None and price >= self.red_strike:
            self.scenario = "above_red"
        else:
            self.scenario = "neutral"

    def check_first_touch(self, price):
        key = price
        self.touch_count[key] = self.touch_count.get(key, 0) + 1
        return self.touch_count[key]

    def generate_signal(self):
        signal = {
            "scenario": self.scenario,
            "current_price": self.current_price,
            "green_strike": self.green_strike,
            "red_strike": self.red_strike,
            "support_strike": self.green_support,
            "resistance_strike": self.red_resistance,
            "reason": [],
        }

        if self.current_price is None:
            signal["action"] = "no_price"
            signal["reason"].append("Current price unavailable")
            return signal

        if self.scenario in ("near_support", "below_support", "below_green"):
            signal["action"] = "BUY"
            signal["recommendation"] = "Consider long near support or pullback zone"
            signal["reason"].append("Price is at or below support zone")
        elif self.scenario in ("near_resistance", "above_resistance", "above_red"):
            signal["action"] = "SELL"
            signal["recommendation"] = "Consider short near resistance or exit longs"
            signal["reason"].append("Price is at or above resistance zone")
        elif self.scenario == "between_support_resistance":
            signal["action"] = "HOLD"
            signal["recommendation"] = "Market is between support and resistance; wait for a clear edge"
            signal["reason"].append("Price is moving inside the range")
        else:
            signal["action"] = "WAIT"
            signal["recommendation"] = "No strong AOC scenario detected"
            signal["reason"].append("Unable to determine actionable scenario")

        if self.scenario == "near_support" and self.check_first_touch(self.current_price) == 1:
            signal["reason"].append("First touch of the support area")

        if self.scenario == "near_resistance" and self.check_first_touch(self.current_price) == 1:
            signal["reason"].append("First touch of the resistance area")

        return signal

    def analyze(self, aoc_data):
        self.load_data(aoc_data)
        self.detect_scenario()
        return self.generate_signal()
''', encoding='utf-8')
print('written')
