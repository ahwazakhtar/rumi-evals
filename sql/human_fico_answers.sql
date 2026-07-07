-- Human coach structured observation answers (ICT Digital Coach stack).
-- NOTE: validated 2026-07-06 that these observations share ZERO audio_url values with
-- RUMI_DB.coaching_sessions — this is a different observation program on a different
-- teacher population. Kept here as the human-reference source for a future paired
-- protocol (human coach scores Rumi recordings) or approximate teacher+date matching.
SELECT
  co.id            AS observation_id,
  DATE(co.created, 'Asia/Karachi') AS observation_date,
  co.coach_id,
  co.template_id,
  oa.question_id,
  q.prompt         AS question_prompt,
  q.is_scored,
  opt.label        AS answer_label,
  opt.value        AS answer_value,
  opt.score_type   AS answer_score_type
FROM tbproddb.coaching_observation co
JOIN tbproddb.coaching_observationanswer oa
  ON oa.observation_id = co.id AND oa.is_active = TRUE
LEFT JOIN tbproddb.coaching_observationquestion q ON q.id = oa.question_id
LEFT JOIN tbproddb.coaching_questionoption opt ON opt.id = oa.single_choice_option_id
WHERE co.is_active = TRUE
