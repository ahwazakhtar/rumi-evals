-- All AI-assigned FICO scores for non-test users, one row per scored session.
-- Feeds: step3a (cross-LLM agreement baseline), step5 (alignment), step6 (longitudinal).
SELECT
  f.id,
  f.user_id,
  f.session_id,
  DATE(f.created_at, 'Asia/Karachi') AS session_date,
  f.audio_duration_seconds,
  f.transcript_length,
  f.is_coaching_transcript,
  f.evaluation_confidence,
  f.overall_score,
  f.B_score, f.C_score, f.D_score, f.F_score,
  f.B1_score, f.B2_score, f.B3_score, f.B4_score, f.B5_score,
  f.B6_score, f.B7_score, f.B8_score, f.B9_score, f.B10_score,
  f.C1_score, f.C2_score, f.C3_score, f.C4_score, f.C5_score, f.C6_score,
  f.C7_score, f.C8_score, f.C9_score, f.C10_score, f.C11_score, f.C12_score,
  f.D1_score, f.D2_score, f.D3_score, f.D4_score, f.D5_score, f.D6_score, f.D7_score,
  f.F1_score, f.F2_score, f.F3_score, f.F4_score, f.F5_score, f.F6_score, f.F7_score, f.F8_score
FROM RUMI_DB.coaching_sessions_with_fico_scores f
JOIN RUMI_DB.users u ON f.user_id = u.id
WHERE u.is_test_user IS NOT TRUE
