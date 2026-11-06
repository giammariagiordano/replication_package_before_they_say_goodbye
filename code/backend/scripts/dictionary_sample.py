"""Random samples for the manual validation of the lexical dictionaries (app/psych.py); see validation/dictionaries.

Run in the api container, with the output redirected to a local file:
  docker compose exec -T api python - < backend/scripts/dictionary_sample.py > dict_sample.json
"""

import json, random, re
from sqlalchemy import select, text
from app.db import session_scope
from app.models import Project
from app.survival import archived_projects
from app.stages.text import clean_text
from app.psych import GRATITUDE, WE, WORD

random.seed(20261102)
with session_scope() as db:
    done = list(db.scalars(select(Project.id).where(Project.status == "done")))
    pids = sorted(set(done) - archived_projects(db, done))
    replies = db.execute(text("""
      select c.id, c.body from comments c tablesample system (2) repeatable (7)
      join aliases ca on ca.id = c.alias_id join developers cd on cd.id = ca.developer_id
      join issues i on i.project_id = c.project_id and i.number = c.issue_number
      join aliases ia on ia.id = i.alias_id join developers od on od.id = ia.developer_id
      where c.project_id = any(:p) and not cd.is_bot and not od.is_bot and cd.id <> od.id"""), {"p": pids}).all()
    msgs = db.execute(text("""
      select 'c'||c.id, c.body from comments c tablesample system (1) repeatable (11)
      join aliases a on a.id = c.alias_id join developers d on d.id = a.developer_id
      where c.project_id = any(:p) and not d.is_bot
      union all
      select 'i'||i.id, i.body from issues i tablesample system (3) repeatable (13)
      join aliases a on a.id = i.alias_id join developers d on d.id = a.developer_id
      where i.project_id = any(:p) and not d.is_bot"""), {"p": pids}).all()
pos, neg = [], []
for cid, body in replies:
    t = clean_text(body)
    if not WORD.search(t.lower()):
        continue
    m = GRATITUDE.search(t)
    (pos if m else neg).append({"id": cid, "text": t[:600], "match": m.group(0) if m else None})
random.shuffle(pos); random.shuffle(neg)
we = []
for mid, body in msgs:
    t = clean_text(body)
    low = t.lower().replace("’", "'")
    hits = [m for m in WORD.finditer(low) if m.group(0) in WE]
    if hits:
        m = random.choice(hits)
        we.append({"id": mid, "word": m.group(0), "context": t[max(0, m.start()-220): m.start()] + "<<" + t[m.start():m.end()] + ">>" + t[m.end(): m.end()+220]})
random.shuffle(we)
print(json.dumps({"n_replies": len(replies), "n_pos": len(pos), "n_neg": len(neg), "n_we": len(we),
                  "grateful": pos[:100], "not_grateful": neg[:100], "we": we[:150]}, ensure_ascii=False))
