// CandidateChip — a compact, clickable candidate reference. Shows the short
// id, its proposal source, and (where known) its validation rank. Clicking
// opens the candidate drawer over whatever view you're on.

import { Show, type Component } from "solid-js";
import { shortCandidateId } from "../models/verdict-model.js";

export interface CandidateChipProps {
  candidateId: string;
  source?: string | null;
  rank?: number | null;
  onClick?: (candidateId: string) => void;
}

export const CandidateChip: Component<CandidateChipProps> = (props) => {
  const label = (): string => shortCandidateId(props.candidateId);
  return (
    <button
      type="button"
      class="tuner-candidate-chip"
      data-testid="candidate-chip"
      onClick={() => props.onClick?.(props.candidateId)}
    >
      <Show when={props.rank != null}>
        <span class="tuner-chip-rank">#{props.rank}</span>
      </Show>
      <span class="tuner-chip-id">{label()}</span>
      <Show when={props.source}>
        <span class="tuner-chip-source">{props.source}</span>
      </Show>
    </button>
  );
};
