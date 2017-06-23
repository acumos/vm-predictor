

from stat import S_ISREG, ST_CTIME, ST_MODE, ST_SIZE, ST_MTIME
import os, sys, time

if __name__ == "__main__": 
    # path to the directory (relative or absolute)
    dirpath = sys.argv[1] if len(sys.argv) == 2 else r'.'

    # get all entries in the directory w/ stats
    entries = (os.path.join(dirpath, fn) for fn in os.listdir(dirpath))
    entries = ((os.stat(path), path) for path in entries)

    # leave only regular files, insert creation date
    entries = ((stat[ST_CTIME], path)  for stat, path in entries if S_ISREG(stat[ST_MODE]))
    #NOTE: on Windows `ST_CTIME` is a creation date 
    #  but on Unix it could be something else
    #NOTE: use `ST_MTIME` to sort by a modification date

    for cdate, path in sorted(entries):
        print time.ctime(cdate), os.path.basename(path)

        

import glob
def cdateSorted (patt):
    # get all entries in the directory w/ stats
    entries = (fn for fn in glob.glob(patt))
    entries = ((os.stat(path), path) for path in entries)

    # leave only regular files, insert creation date
    #entries = ((stat[ST_CTIME], path, stat[ST_SIZE])  for stat, path in entries if S_ISREG(stat[ST_MODE]))
    entries = ((stat[ST_MTIME], path, stat[ST_SIZE])  for stat, path in entries if S_ISREG(stat[ST_MODE]))
    #NOTE: on Windows `ST_CTIME` is a creation date 
    #  but on Unix it could be something else
    #NOTE: use `ST_MTIME` to sort by a modification date

    #for cdate, path in sorted(entries):
    #    print time.ctime(cdate), os.path.basename(path)
    return sorted(entries)


def mdateFltSorted (patt):
    entries = (fn for fn in glob.glob(patt))
    entries = ((os.stat(path), path) for path in entries)

    # leave only regular files, insert creation date
    #entries = ((stat[ST_CTIME], path, stat[ST_SIZE])  for stat, path in entries if S_ISREG(stat[ST_MODE]))
    entries = ((os.path.getmtime(path), path, stat[ST_SIZE])  for stat, path in entries if S_ISREG(stat[ST_MODE]))
    #NOTE: on Windows `ST_CTIME` is a creation date 
    #  but on Unix it could be something else
    #NOTE: use `ST_MTIME` to sort by a modification date

    #for cdate, path in sorted(entries):
    #    print time.ctime(cdate), os.path.basename(path)
    return sorted(entries)



# Example: './tmp/Segment_20140708_122618_707_10520.mov'
from seg import segNameToTime

def namedateSorted (patt):
    files = glob.glob(patt)
    entries = []
    for path in files:
        try:
            sz = os.stat(path)[ST_SIZE]
            t, dur = segNameToTime (path)
            entry = t, path, sz, dur
            entries.append (entry)

        except Exception as e:
            print "Exception encountered for path", path, ":", e
            
    return sorted(entries)





    