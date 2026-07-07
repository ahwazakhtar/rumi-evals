-- Completed sessions with transcript + analysis payloads for LLM-judge evals
-- (step 3b hallucination spot-check, step 4a feedback-quality proxy, step 3a re-scoring).
SELECT
  cs.id,
  cs.user_id,
  DATE(SAFE_CAST(cs.created_at AS TIMESTAMP), 'Asia/Karachi') AS session_date,
  cs.transcript_language,
  cs.transcript_text,
  cs.analysis_data,
  cs.prioritized_action
FROM RUMI_DB.coaching_sessions cs
JOIN RUMI_DB.users u ON cs.user_id = u.id
WHERE u.is_test_user IS NOT TRUE
  AND cs.status = 'completed'
  AND cs.transcript_text IS NOT NULL
  AND LENGTH(cs.transcript_text) > 200
