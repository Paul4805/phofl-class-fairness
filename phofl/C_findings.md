# C - correlated participation

Under linear: FM=0.1439, min=0.2650, rho(p,acc)=-0.758.  Under entropy_correlated: FM=0.1805, min=0.1750, rho(p,acc)=+0.721.  The entropy pattern raises mean accuracy slightly but significantly increases per-client dispersion (p=0.041) and collapses the tail; the linear-vs-entropy mean-accuracy test is not significant (p=0.410).  This is a fairness/tail effect, not the hypothesized mean-advantage collapse.  Escalate C to CIFAR-10/half-scale if a tail-focused claim is the goal.
