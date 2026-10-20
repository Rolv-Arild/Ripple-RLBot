"""Frame-based decision cadence; late packets never start a catch-up burst."""


class DecisionClock:
    def __init__(self, tick_skip=4):
        self.tick_skip = tick_skip
        self.next_frame = None
        self.last_frame = None

    def advance(self, frame):
        if self.next_frame is None or frame < self.last_frame:
            self.next_frame = frame
        self.last_frame = frame
        if frame < self.next_frame:
            return 0
        periods = (frame - self.next_frame) // self.tick_skip + 1
        self.next_frame += periods * self.tick_skip
        return periods
