from __future__ import print_function

import cherrypy
import os
import subprocess
import shlex
import urllib
import random
import string


thumbBase = "./thumbs/"

Running = True

def thumbCleaner(dir):
    import dirList
    import time
    maxlife = 24 * 60 * 60  # one day
    frequency = 3600        # one hour
    lastcheck = 0
    while Running:
        now = time.time()
        if now - lastcheck >= frequency:
            print ("thumbCleaner starting.")
            thumbs = dirList.cdateSorted(dir + "/*.jpg")
            for ctime, name, sz in thumbs:
                if now - ctime > maxlife:      
                    print ("thumbCleaner: expired -- ", name)
                    try:                    # because there may be multiple thumbCleaners
                        os.remove(name)
                    except:
                        pass
            print ("thumbCleaner sleeping.")
            lastcheck = time.time()
        time.sleep(2)


def randomString():
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))


def processCommand(cmdstring, dir='.', verbose=False):
    if verbose:
        print ("in dir='%s' Running: %s" % (dir, cmdstring))
    cmds = shlex.split(cmdstring)
    proc = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False, cwd=dir)
    (out, err) = proc.communicate()
    if err:
        print ("Listener:  Error", err, "running command:")
        print ("          ", cmd)
        out = "ERROR: " + str(err)
    return err, out


def getMovieFrame (srcVid):
    basename = ".".join(srcVid.split(".")[:-1])
    cachename = thumbBase + urllib.quote_plus(basename) + ".jpg"
    if not os.path.isfile (cachename):
        tmpname = randomString() + ".jpg"
        commandText = 'bash -c \"source /var/tashkent/cae/setCAEenv.sh; ffmpeg -i %s -r 1 -vframes 1 %s"' % (srcVid, tmpname)
        processCommand(commandText)
        processCommand("mv %s %s" % (tmpname, cachename))
    return cachename
    

def readCSV (filename, sep=','):
    import csv
    fp = open(filename)
    if sep=='pipe':
        read = csv.reader(fp, delimiter='|')
    else:
        read = csv.reader(fp)
    return [row for row in read]
    
    

class HelloWorld(object):
    def index(self):
        return "Hello World!"
    index.exposed = True

   
    def fetch(self, file):
        f = open(file, "rb")
        return f.read()
    fetch.exposed = True

    
    def text(self, file):
        f = open(file, "r")
        data = f.read()
        data = data.replace ("<", "&lt;")
        data = data.replace (">", "&gt;")
        HTM = '<!DOCTYPE html>\n'
        HTM += '<html>\n'
        HTM += '<pre>'
        HTM += data
        HTM += '</pre>'
        return HTM
    text.exposed = True

    
    def pretty_csv(self, file, sep=','):
        table = readCSV (file, sep)
        HTM = '<!DOCTYPE html>\n'
        HTM += '<html>\n'
        HTM += '<table border="1" style="width:100%">'
        for row in table:
            HTM += "<tr>\n"
            for item in row:
                HTM += "<td>" + str(item) + "</td>\n"
            HTM += "</tr>\n"
        HTM += '</table>'
        return HTM
    pretty_csv.exposed = True
        

    def code(self, file):
        f = open(file, "r")
        data = f.read()
        data = data.replace ("<", "&lt;")
        data = data.replace (">", "&gt;")
        HTM = '<!DOCTYPE html>\n'
        HTM += '<html>\n'
        HTM += '<script src="https://google-code-prettify.googlecode.com/svn/loader/run_prettify.js"></script>'
        HTM += '<style> .prettyprint ol.linenums > li { list-style-type: decimal; } </style>'
        HTM += '<pre class="prettyprint linenums">'
        HTM += data
        HTM += '</pre>'
        return HTM
    code.exposed = True

    
    def thumb(self, mov):
        tname = getMovieFrame(mov)
        f = open(tname, "r")
        return f.read()
    thumb.exposed = True

    
    def show(self, file):
        HTM = '<!DOCTYPE html>\n'
        HTM += '<html>\n'
        HTM += '<head>\n'
        HTM += '<img src=/fetch?file=%s>' % file
        return HTM
    show.exposed = True
    
    
    def testing(self):
        HTM = '<!DOCTYPE html>\n'
        HTM += '<html>\n'
        HTM += '<head>\n'
        HTM += '<img src="https://cdn2.iconfinder.com/data/icons/windows-8-metro-style/512/document.png">'
        return HTM
    testing.exposed = True

    
    def browse(self, dir="*", width="160", height="120"):
        import glob

        # fix relative path
        dir = dir.strip()
        if dir[0] != '.' and dir[0] != '/':
            dir = "./" + dir
            
        # separate fixed part from wild part (could be "./*/*")
        sdirs = dir.split("/")
        firstwild = -1
        for idx,sd in enumerate(sdirs):
            if '*' in sd:
                firstwild = idx
                break
        if firstwild < 0:
            if dir[-1] != "/":
                dir += "/"
                firstwild = len(sdirs)
            else:
                firstwild = len(sdirs) - 1
            dir += "*"
        fix = "/".join(sdirs[0:firstwild])
        fixed_part = len(fix) + 1
        
        listing = sorted(glob.glob(dir))
        HTM = DivHead
        HTM += '<body>\n'
        for path in listing:           
            fname = path[fixed_part:]
            parts = fname.split('.')
            if os.path.isdir(path):
                HTM += '<div class="img"><a href="/browse?dir=%s"><img src="%s" width="%s" height="%s"></a><div class="desc">%s</div></div>' % (path,FolderIcon,width,height,fname)
            elif parts[-1] == "jpg" or parts[-1] == "png":
                HTM += '<div class="img"><a href="/show?file=%s"><img src="/fetch?file=%s" width="%s" height="%s"></a><div class="desc">%s</div></div>' % (path,path,width,height,fname)
            elif parts[-1] == "mp4" or parts[-1] == "mov":
                HTM += '<div class="img"><video width="%s" height="%s" controls preload="none" poster="/thumb?mov=%s"><source src="/fetch?file=%s" type="video/mp4"></video><div class="desc">%s</div></div>' % (width, height, path, path, fname)
            elif parts[-1] == "py" or parts[-1] == "cpp" or parts[-1] == "c" or parts[-1] == "h" or parts[-1] == "hpp":
                HTM += '<div class="img"><a href="/code?file=%s"><img src="%s" width="%s" height="%s"></a><div class="desc">%s</div></div>' % (path,CodeIcon,width,height,fname)
            elif parts[-1] == "csv":
                HTM += '<div class="img"><a href="/pretty_csv?file=%s"><img src="%s" width="%s" height="%s"></a><div class="desc">%s</div></div>' % (path,CSVIcon,width,height,fname)
            elif parts[-1] == "psv":
                HTM += '<div class="img"><a href="/pretty_csv?file=%s&sep=pipe"><img src="%s" width="%s" height="%s"></a><div class="desc">%s</div></div>' % (path,TableIcon,width,height,fname)
            else:
                HTM += '<div class="img"><a href="/text?file=%s"><img src="%s" width="%s" height="%s"></a><div class="desc">%s</div></div>' % (path,DocumentIcon,width,height,fname)
        HTM += '</body>\n'
        HTM += '</html>\n'
        return HTM
    browse.exposed = True
    

    
#DocumentIcon = 'https://cdn2.iconfinder.com/data/icons/windows-8-metro-style/512/document.png'
#DocumentIcon = 'http://www.iconshock.com/img_jpg/LUMINA/general/jpg/256/document_icon.jpg'
#DocumentIcon = 'http://www.vectors4all.net/preview/ronoaldo-new-document-clip-art.jpg'
DocumentIcon = 'http://www.edutech.nodak.edu/chromebooks/files/2014/09/google_docs.png'
#FolderIcon = 'http://png-3.findicons.com/files/icons/727/leopard/128/folder.png'
FolderIcon = 'http://fc06.deviantart.net/fs70/f/2012/292/a/8/steampunk_victorian_folder_icon_by_pendragon1966-d5i99js.png'
#CodeIcon = 'http://www.pubzi.com/f/source-code-icon.svg'
#CodeIcon = 'http://findicons.com/files/icons/1714/dropline_neu/128/text_x_source.png'
CodeIcon = 'https://www.softlanding.com/updates/concrete5.6.3.4/concrete/images/icons/filetypes/zip.png'
CSVIcon = 'http://www.colabrativ.com/images/OxygenTeam_speadsheet+CSV_128x128.png'
TableIcon = 'http://www.rocketroute.com/wp-content/uploads/mimetypes_office_spreadsheet.png'

    
DivHead = '''
<!DOCTYPE html>
<html>
<head>
<style>
div.img {
    margin: 5px;
    padding: 5px;
    border: 1px solid #0000ff;
    height: auto;
    width: auto;
    float: left;
    text-align: center;
}	

div.img img {
    display: inline;
    margin: 5px;
    border: 1px solid #ffffff;
}

div.img a:hover img {
    border: 1px solid #0000ff;
}

div.desc {
  text-align: center;
  font-weight: normal;
  width: 120px;
  margin: 5px;
  word-wrap: break-word;
}
</style>
</head>
'''
    
    
    
    
    
cleanThread = None    
    
def stopAll ():
    global Running
    Running = False
stopAll.priority = 10
    
    
if __name__ == "__main__":    
    import threading
    cleanThread = threading.Thread(target = thumbCleaner, args = (thumbBase,))
    cleanThread.start()
    
    
    cherrypy.engine.subscribe('stop', stopAll)
    
    cherrypy.server.socket_host = "0.0.0.0"
    cherrypy.server.socket_port = 8084
        
    cherrypy.quickstart(HelloWorld())



