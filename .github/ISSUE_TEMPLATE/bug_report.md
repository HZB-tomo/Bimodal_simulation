---
name: Bug report
about: Something produces the wrong result or crashes
labels: bug
---

**What happened**

**What you expected**

**Minimal reproduction**

```python
from neutron_xray_sim import ...
```

**Environment**

```
python -c "import neutron_xray_sim, numpy, scipy, sys; print(neutron_xray_sim.__version__, numpy.__version__, scipy.__version__, sys.version)"
python -c "from neutron_xray_sim.reconstructor import _astra_ok; print('ASTRA:', _astra_ok())"
```

**Physics or code?** If a number looks wrong rather than the code crashing,
please paste the output of `MATERIALS.audit()` too.
