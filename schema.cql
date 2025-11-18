// === Uniqueness constraints ===
CREATE CONSTRAINT IF NOT EXISTS FOR (i:Item)          REQUIRE i.answer_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (q:Question)      REQUIRE q.question_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (g:Gold)          REQUIRE g.answer_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (m:Model)         REQUIRE m.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (p:Prompt)        REQUIRE p.prompt_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (c:Call)          REQUIRE c.call_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (a:Answer)        REQUIRE a.answer_uid IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (j:Judgment)      REQUIRE j.judgment_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (r:Run)           REQUIRE r.run_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (x:Misconception) REQUIRE x.text IS UNIQUE;

// === Helpful indexes ===
CREATE INDEX IF NOT EXISTS FOR (p:Prompt)  ON (p.kind, p.attack_label);
CREATE INDEX IF NOT EXISTS FOR (a:Answer)  ON (a.condition, a.attack_label);
CREATE INDEX IF NOT EXISTS FOR (m:Model)   ON (m.role);
