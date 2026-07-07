-- Session-level pipeline outcomes for the Step 1 audio-capture gate.
SELECT
  cs.id,
  cs.user_id,
  DATE(SAFE_CAST(cs.created_at AS TIMESTAMP), 'Asia/Karachi') AS session_date,
  cs.status,
  cs.failed_step,
  cs.error_message,
  cs.audio_duration_seconds,
  cs.audio_format,
  cs.audio_size_bytes,
  cs.transcript_language,
  cs.diarization_confidence,
  LENGTH(cs.transcript_text) AS transcript_chars,
  qm.retry_count,
  qm.had_errors,
  qm.processing_time_seconds
FROM RUMI_DB.coaching_sessions cs
JOIN RUMI_DB.users u ON cs.user_id = u.id
LEFT JOIN RUMI_DB.coaching_quality_metrics qm ON qm.coaching_session_id = cs.id
WHERE u.is_test_user IS NOT TRUE
  AND cs.status != 'test_cleanup'
