/** Plain labels for the call view. Internal keys remain available in raw JSON. */

const CHOICES: Record<string, string> = {
  ev: "electric car",
  suv: "SUV",
  not_stated: "not stated yet",
  new_vehicle: "new vehicle visit",
  used_vehicle: "used vehicle visit",
  general_question: "general question",
  wrong_number: "wrong number",
  none_of_these: "none of the offered times",
  slot_1: "first offered time",
  slot_2: "second offered time",
  slot_3: "third offered time",
};

const ACTIONS: Record<string, string> = {
  ask_vehicle: "Ask what vehicle they have",
  ask_location: "Ask which location works",
  ask_time: "Ask when they can visit",
  ask_alternative_time: "Ask for another day",
  ask_detail: "Ask for more detail",
  confirm_booking: "Check appointment times",
  offer_slots: "Offer available times",
  ask_which_slot: "Clarify which time works",
  booked: "File the appointment",
  answer_hours: "Answer from store hours",
  close_wrong_number: "End the wrong-number call",
  offer_transfer: "Connect to a person",
};

const ENDINGS: Record<string, string> = {
  booked: "Appointment booked",
  dispatched: "Roadside help requested",
  transferred: "Connected to a team",
  answered: "Question answered",
  closed: "Call ended",
  awaiting_caller: "Waiting for caller",
};

const BINARY: Record<string, string> = {
  is_safe_to_drive: "Unsafe to drive",
  needs_human: "Needs a person",
  changed: "New information",
};

export function choiceLabel(value: string | null | undefined): string {
  if (!value) return "—";
  return CHOICES[value] ?? value.replace(/_/g, " ");
}

export function actionLabel(value: string | null | undefined): string {
  if (!value) return "—";
  return ACTIONS[value] ?? choiceLabel(value);
}

export function endingLabel(value: string | null | undefined): string {
  if (!value) return "Call complete";
  return ENDINGS[value] ?? choiceLabel(value);
}

export function binaryLabel(key: string): string {
  return BINARY[key] ?? "Yes";
}

export function handlerLabel(value: string): string {
  return value === "auto" ? "Handled here" : value === "human" ? "Person needed" : choiceLabel(value);
}

export function policyValue(value: unknown): string {
  if (Array.isArray(value)) return value.length ? value.map((item) => choiceLabel(String(item))).join(", ") : "Nothing missing";
  if (typeof value === "string") return actionLabel(value);
  return value == null ? "—" : String(value);
}
