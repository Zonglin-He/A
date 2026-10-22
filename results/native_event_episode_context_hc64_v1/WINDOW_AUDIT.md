# Postseal episode coverage and case audit

Window coverage below uses GT only AFTER prediction seal/scoring. No window or query is removed or selected using these values. A shifted window may still cover the actual event.

|Offline cohort|N|episode event coverage|shifted coverage|t gain vs global clause pp|t gain vs Source pp|
|---|---:|---:|---:|---:|---:|
|supported52|52|0.9613|0.7547|-0.3415|+3.2400|
|window_covers_event|40|1.0000|0.7242|+1.1702|+2.9182|
|window_cuts_event|12|0.8321|0.8565|-5.3807|+4.3125|
|short|10|1.0000|0.8892|+1.5385|-6.9841|
|long|42|0.9520|0.7227|-0.7891|+5.6743|

## Concrete same-query contrasts

|Query|episode coverage|t change vs global clause pp|t change vs Source pp|
|---|---:|---:|---:|
|RC019|0.6654|-14.7619|+9.8821|
|RC052|0.7567|-14.0271|+0.0000|
|RC025|0.8491|-11.7725|+0.0000|
|RC029|0.8364|-7.2150|+18.4668|
|RC054|1.0000|+7.0172|+0.1449|
|RC015|1.0000|+7.5958|+17.0373|
|RC014|1.0000|+9.1154|-34.7271|
|RC016|1.0000|+11.4978|+16.4305|

Coverage and case associations do not prove cropping is the sole cause: both fusion context and available interval support change. Short-event length summaries separate expansion from mere shifted-context quality. No GT-selected fallback is adopted. [Complete numeric audit](window_audit.json).
