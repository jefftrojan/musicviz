create table if not exists public.study_responses (
  id uuid primary key default gen_random_uuid(),
  participant_id text not null check (participant_id ~ '^[0-9a-f]{12}$'),
  pair_id text not null check (pair_id in (
    'bg_calm_shift', 'bg_shuffle', 'sg_calm_random',
    'sg_heavy_shift', 'sd_shuffle', 'sd_heavy_random'
  )),
  backbone text not null check (backbone in ('biggan', 'stylegan2', 'sd')),
  song text not null check (song in ('calm_blue_lake', 'acoustic_shuffle', 'heavy_drums_bass')),
  seed integer not null check (seed between 0 and 2),
  control_type text not null check (control_type in ('shifted', 'shuffled', 'matched_random')),
  aligned_position text not null check (aligned_position in ('A', 'B')),
  presentation_order integer not null check (presentation_order between 1 and 6),
  selected_clip text not null check (selected_clip in ('A', 'B', 'no_difference')),
  aligned_selected numeric not null check (aligned_selected in (0, 0.5, 1)),
  attention_pass boolean not null,
  audio_check_pass boolean not null,
  response_time_seconds integer not null check (response_time_seconds >= 0),
  comment text not null default '' check (char_length(comment) <= 600),
  sync_aligned smallint not null check (sync_aligned between 1 and 5),
  sync_control smallint not null check (sync_control between 1 and 5),
  smooth_aligned smallint not null check (smooth_aligned between 1 and 5),
  smooth_control smallint not null check (smooth_control between 1 and 5),
  coherence_aligned smallint not null check (coherence_aligned between 1 and 5),
  coherence_control smallint not null check (coherence_control between 1 and 5),
  appeal_aligned smallint not null check (appeal_aligned between 1 and 5),
  appeal_control smallint not null check (appeal_control between 1 and 5),
  age_range text not null check (age_range in (
    '18–24', '25–34', '35–44', '45–54', '55+', 'Prefer not to say'
  )),
  experience text not null default '',
  final_comment text not null default '' check (char_length(final_comment) <= 800),
  study_id text not null check (study_id = 'musicviz-causal-pairs-v1'),
  started_at timestamptz not null,
  ended_at timestamptz not null,
  total_duration_seconds integer not null check (total_duration_seconds >= 0),
  received_at timestamptz not null default now(),
  unique (study_id, participant_id, pair_id)
);

alter table public.study_responses enable row level security;

revoke all on table public.study_responses from anon, authenticated;
grant insert on table public.study_responses to anon;

drop policy if exists "anonymous study submissions" on public.study_responses;
create policy "anonymous study submissions"
on public.study_responses
for insert
to anon
with check (
  study_id = 'musicviz-causal-pairs-v1'
  and attention_pass is true
  and audio_check_pass is true
);

comment on table public.study_responses is
  'Anonymous trial-level responses for the MusicViz paired-comparison study.';
