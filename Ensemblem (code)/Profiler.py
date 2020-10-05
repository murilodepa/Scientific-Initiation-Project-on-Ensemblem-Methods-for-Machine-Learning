import time
import sys

class Milestones:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    PRFH = '[' + OKBLUE + 'Profiler' + ENDC + ']'
    
    def __init__(self, ignore_start=False):
        print('[Profiler starting...]', file=sys.stderr)
        if ignore_start:
            self.milestones = {}
        else:
            self.milestones = {int(time.time() * 1000):"Start"}
        self.ign_st = ignore_start
        
    def colortxt(self, color, txt):
        return color + txt + self.ENDC
    
    def add_milestone(self, name):
        now = int(time.time() * 1000)
        ok = False
        try:
            while not ok:
                temp = self.milestones[now]
                now = now + 1
        except KeyError:
            ok = True
            pass
        self.milestones[now] = name
        print('{} @ {}: {}'.format(self.PRFH, self.build_timeview(time.localtime(now/1000.0), s_date=False), name), file=sys.stderr)
            
        
    def exhibit(self, show_raw_time=False, show_local_time=True, show_deltas=True, max_d=None):
        timestamps = [0]
        time_diffs = [0]
        for key in self.milestones:
            timestamps.append(key)
            time_diffs.append(key - timestamps[-2])
        timestamps = timestamps[1:]
        time_diffs = time_diffs[2:]
        #print("Timestamps: " + str(timestamps))
        #print("Time_diffs: " + str(time_diffs))
        scale = min(time_diffs)
        max_t = max(time_diffs)
        if max_d == None:
            max_d = max_t
        n = 0
        output = "0"
        for key in self.milestones:
            try:
                length = int(self.map_val(scale, time_diffs[n], max_t, 1, max_d))
                #length = int(time_diffs[n] / float(scale))
                n += 1
                output = output + ('=' * length) + str(n)
            except IndexError:
                pass
        print(output)
        n = 0
        for key in self.milestones:
            msg = "%3d: %s" % (n, self.milestones[key])
            if show_raw_time:
                msg += " - %dms" % (key)
            if show_local_time:
                msg += " - %s" % (self.build_timeview(time.localtime(key/1000.0)))
            if show_deltas:
                if (n > 0):
                    try:
                        msg += " - %.2fs" % (time_diffs[n - 1] / 1000.0)
                    except IndexError:
                        pass
            print(msg)
            n += 1
            
    def build_timeview(self, ts, s_date=True):
        if s_date:
            return "%02d/%02d/%4d - %02d:%02d:%02d" % (ts.tm_mday, ts.tm_mon, ts.tm_year, ts.tm_hour, ts.tm_min, ts.tm_sec)
        else:
            return "%02d:%02d:%02d" % (ts.tm_hour, ts.tm_min, ts.tm_sec)
        
    def map_val(self, xmin, xval, xmax, ymin, ymax):
        dx = xmax - xmin
        dy = ymax - ymin
        ddx = xval - xmin
        yval = ymin + dy * ddx / dx
        return yval
