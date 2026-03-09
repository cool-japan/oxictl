# oxictl::safety

Functional safety subsystem for real-time control.

## Key Types

- **`Watchdog<S>`** - Software watchdog timer (kick or trip)
- **`FaultSeverity`** - Info, Warning, Error, Critical
- **`FaultHandler<N>`** - Collects faults in a heapless buffer, tracks worst severity
- **`SafetyMonitor<S, N, M, T>`** - Aggregates range, rate, and timeout monitors
- **Monitors**: `RangeMonitor`, `RateMonitor`, `TimeoutMonitor`

## Safety Responses

| Severity | Default Response |
|----------|-----------------|
| Info     | Log & continue  |
| Warning  | Log & continue  |
| Error    | Degrade         |
| Critical | Emergency stop  |

## no_std Compatibility

Fully `no_std` + `no_alloc`. Uses const generics and `heapless::Vec` for fixed-size storage.
