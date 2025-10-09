def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def ttc(ego_v, npc_v, distance, eps=1e-3):
    rel_v = max(ego_v - npc_v, eps)
    return distance / rel_v
