/**
 * WaveTrader Market Session Status
 * Updates forex session indicators in the navbar based on current UTC time.
 *
 * Sessions (UTC):
 *   Sydney:   22:00 – 07:00
 *   Tokyo:    00:00 – 09:00
 *   London:   07:00 – 16:00
 *   New York: 12:00 – 21:00
 *   Forex:    Sun 22:00 – Fri 22:00
 */

function updateMarketSessions() {
    const now = new Date();
    const utcH = now.getUTCHours();
    const utcM = now.getUTCMinutes();
    const day = now.getUTCDay(); // 0=Sun, 6=Sat
    const t = utcH + utcM / 60;  // decimal hours

    // Forex market: open Sun 22:00 UTC → Fri 22:00 UTC
    const forexOpen = !(
        day === 6 ||                        // Saturday
        (day === 0 && t < 22) ||            // Sunday before 22:00
        (day === 5 && t >= 22)              // Friday after 22:00
    );

    // Sydney: 22:00 – 07:00 UTC (wraps midnight)
    const sydneyOpen = forexOpen && (t >= 22 || t < 7);

    // Tokyo: 00:00 – 09:00 UTC
    const tokyoOpen = forexOpen && (t >= 0 && t < 9);

    // London: 07:00 – 16:00 UTC
    const londonOpen = forexOpen && (t >= 7 && t < 16);

    // New York: 12:00 – 21:00 UTC
    const nyOpen = forexOpen && (t >= 12 && t < 21);

    setSessionStatus('forex', forexOpen);
    setSessionStatus('sydney', sydneyOpen);
    setSessionStatus('tokyo', tokyoOpen);
    setSessionStatus('london', londonOpen);
    setSessionStatus('newyork', nyOpen);
}

function setSessionStatus(id, isOpen) {
    const dot = document.getElementById(`session-${id}-dot`);
    const session = document.getElementById(`session-${id}`);
    if (dot) {
        dot.className = 'wt-session-dot ' + (isOpen ? 'open' : 'closed');
    }
    if (session) {
        session.classList.toggle('active', isOpen);
    }
}

// Run immediately and update every 30 seconds
document.addEventListener('DOMContentLoaded', () => {
    updateMarketSessions();
    setInterval(updateMarketSessions, 30000);
});
