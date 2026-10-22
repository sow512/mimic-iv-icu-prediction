#!/usr/bin/env python3
"""Streaming MBOX -> SQLite index; preserves source offsets for later extraction."""
import argparse, sqlite3, os, re, json, email
from email import policy
from email.parser import BytesHeaderParser
from datetime import datetime

KEYWORDS = {
 'placement': r'placement|campus recruit|campus driv|on.campus|pre.placement|\bctc\b|\blpa\b|eligibility criteria|recruitment driv|job description|\bppo\b',
 'internship': r'internship|intern opening|summer intern|winter intern|industrial training|stipend',
 'jobs': r'job opening|job opportunit|hiring|vacanc|career opportunit|graduate trainee|management trainee|analyst role|walk.in interview',
 'conference': r'conference|symposium|congress|call for papers|abstract submission|research meet',
 'programs': r'fellowship|scholarship|summer school|workshop|hackathon|competition|case compet|bootcamp|exchange program|research opportunit',
}
COMPILED = {k:re.compile(v,re.I) for k,v in KEYWORDS.items()}

def decode_header(value):
 if not value: return ''
 try:
  from email.header import decode_header as dh
  return ''.join((b.decode(enc or 'utf-8','replace') if isinstance(b,bytes) else b) for b,enc in dh(str(value)))
 except Exception: return str(value)

def iter_mbox(path):
 """Read messages by offsets, never loading whole archive into memory."""
 with open(path,'rb') as f:
  start=0; pos=0; current=None
  while True:
   line=f.readline()
   if not line:
    if current is not None: yield current,pos
    break
   if line.startswith(b'From '):
    if current is not None: yield current,pos
    current=pos
   pos=f.tell()

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('mbox'); ap.add_argument('--db',default='mail_opportunities.sqlite'); ap.add_argument('--limit',type=int,default=0)
 args=ap.parse_args()
 con=sqlite3.connect(args.db)
 con.execute('PRAGMA journal_mode=WAL')
 con.execute('CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, start_byte INTEGER UNIQUE, end_byte INTEGER, date TEXT, sender TEXT, recipients TEXT, subject TEXT, message_id TEXT, labels TEXT, categories TEXT, attachment_names TEXT, body_preview TEXT)')
 con.execute('CREATE INDEX IF NOT EXISTS idx_categories ON messages(categories)')
 con.execute('CREATE INDEX IF NOT EXISTS idx_date ON messages(date)')
 con.execute('CREATE VIRTUAL TABLE IF NOT EXISTS mail_search USING fts5(subject, sender, body_preview, content="messages", content_rowid="id")')
 count=0
 with open(args.mbox,'rb') as f:
  for start,end in iter_mbox(args.mbox):
   if args.limit and count>=args.limit: break
   f.seek(start)
   # parse complete message one at a time; large attachment messages may use RAM
   raw=f.read(end-start)
   if raw.startswith(b'From '): raw=raw.partition(b'\n')[2]
   try:
    msg=email.message_from_bytes(raw,policy=policy.default)
    subject=decode_header(msg.get('Subject'))
    sender=decode_header(msg.get('From'))
    recipients=decode_header(msg.get('To'))
    date=str(msg.get('Date',''))
    mid=str(msg.get('Message-ID',''))
    labels=str(msg.get('X-Gmail-Labels',''))
    texts=[]; attachments=[]; used=0
    for part in msg.walk():
     filename=part.get_filename()
     if filename: attachments.append(decode_header(filename))
     if part.get_content_type() not in ('text/plain','text/html') or part.get_content_disposition()=='attachment': continue
     try:
      s=part.get_content()
      if not isinstance(s,str): continue
     except Exception: continue
     if part.get_content_type()=='text/html':
      s=re.sub(r'<[^>]*>',' ',s)
     if used<12000:
      texts.append(s[:12000-used]); used+=len(texts[-1])
    body=' '.join(texts)
    haystack=' '.join([subject,sender,labels,' '.join(attachments),body])
    cats=','.join(k for k,p in COMPILED.items() if p.search(haystack))
    cur=con.execute('INSERT OR IGNORE INTO messages(start_byte,end_byte,date,sender,recipients,subject,message_id,labels,categories,attachment_names,body_preview) VALUES (?,?,?,?,?,?,?,?,?,?,?)',(start,end,date,sender,recipients,subject,mid,labels,cats,json.dumps(attachments),body[:12000]))
    if cur.rowcount:
     rid=cur.lastrowid
     con.execute('INSERT INTO mail_search(rowid,subject,sender,body_preview) VALUES (?,?,?,?)',(rid,subject,sender,body[:12000]))
   except Exception as exc:
    print('Could not parse at byte',start,repr(exc),flush=True)
   count+=1
   if count%1000==0:
    con.commit(); print('Processed',count,'messages',flush=True)
 con.commit()
 print('Finished. Scanned:',count,'Database:',os.path.abspath(args.db))
 for cat,n in con.execute("SELECT CASE WHEN categories='' THEN 'unclassified' ELSE 'potentially relevant' END, COUNT(*) FROM messages GROUP BY 1"):
  print(cat,n)
 con.close()
if __name__=='__main__': main()
