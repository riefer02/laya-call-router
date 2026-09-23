import { useRun } from "../store/run";
import { choiceLabel } from "../copy";

/** "2026-09-22" + "08:00" -> "Tuesday 22 September, 8am". Spoken form, because it was spoken. */
function spoken(day: string, time: string): string {
  const [hh, mm] = time.split(":").map(Number);
  const suffix = hh < 12 ? "am" : "pm";
  const hour = hh % 12 || 12;
  const clock = `${hour}${mm ? `:${String(mm).padStart(2, "0")}` : ""}${suffix}`;
  try {
    const d = new Date(`${day}T00:00:00`);
    const date = d.toLocaleDateString(undefined, {
      weekday: "long",
      day: "numeric",
      month: "long",
    });
    return `${date}, ${clock}`;
  } catch {
    return `${day} ${clock}`;
  }
}

export default function BookingCard() {
  const booking = useRun((s) => s.summary?.booking ?? null);

  if (!booking) return null;

  return (
    <div className="border-t border-emerald-900/60 bg-emerald-950/30 px-4 py-2.5">
      <div className="flex items-center gap-2">
        <span className="text-emerald-400">✓</span>
        <span className="text-[11px] font-semibold uppercase tracking-wider text-emerald-300">
          appointment filed
        </span>
        <span className="font-mono text-[10px] text-emerald-600">{booking.id}</span>
        <span className="ml-auto font-mono text-[10px] text-slate-500">
          {booking.duration_min} min
        </span>
      </div>

      <div className="mt-1.5 flex flex-wrap items-baseline gap-x-4 gap-y-1 text-[12px]">
        <span className="text-slate-100">
          <span className="text-slate-400">what </span>
          {choiceLabel(booking.subqueue ?? booking.destination)}
        </span>
        <span className="text-slate-100">
          <span className="text-slate-400">where </span>
          {choiceLabel(booking.location)}
        </span>
        <span className="font-semibold text-emerald-200">
          {spoken(booking.slot_day, booking.slot_time)}
        </span>
        {booking.caller_name && (
          <span className="text-slate-100">
            <span className="text-slate-400">who </span>
            {booking.caller_name}
          </span>
        )}
        {booking.callback_number && (
          <span className="font-mono text-[11px] text-slate-300">{booking.callback_number}</span>
        )}
        {booking.vehicle && (
          <span className="text-slate-400">
            <span className="text-slate-500">vehicle </span>
            {choiceLabel(booking.vehicle)}
          </span>
        )}
        <span className="ml-auto text-slate-500">
          team <span className="text-amber-300">{booking.queue}</span>
        </span>
      </div>
    </div>
  );
}
