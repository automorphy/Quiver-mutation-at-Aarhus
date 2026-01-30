# {'accuracy': 1.0, 'average_score': 0.13892360498988424, 'neg_code_length': -0.5657503714710252, 'neg_undetermined_fraction': -0.017478813559322032}

def decision_tree(x):
  A, B, C = x[0] * x[5], x[1] * x[4], x[2] * x[3]
  p2 = (x[1]*x[5] - x[3]*x[4] + x[0]*x[4] - x[1]*x[3] + x[3]*x[5] - x[0]*x[2]
        - x[1]*x[2] - x[0]*x[1] + x[0]*x[3] - x[2]*x[5] - x[4]*x[5] - x[2]*x[4])
  p6 = A * B * C * (B - A - C)

  if p6 > 8 or p6 == 3: return 1
  if p6 == -1: return -1

  if p6 != 0:
    if p6 == -16:
      if p2 in {4, 13, 16}: return 1
      if p2 in {-16, -13, -11, -8, -7, -5, -4, 5, 7, 11}: return -1
      p4 = A*C - A*B - B*C
      return 1 if p4 == -1 else -1 if p4 == -16 else None

    d = { -256: ({16}, {-16}), -192: ({12}, {-12}), -112: ({8}, {-8}),
          -40: ({2, 8, 10}, {-10, -8, -2}),
          -64: ({13, 16}, {-16, -13, -11, -8, -5, 5, 8, 11}),
          -24: ({12}, {-12, -6, 6}), -12: ({9}, {-9, -3, 3}),
          -4: ({8}, {-11, -8, -7, -5, -4, 4, 5, 7, 11}),
          8: ({-2, 2, 14}, {-14, -10, -8, 8, 10}) }
    if p6 in d:
      p, n = d[p6]
      return 1 if p2 in p else -1 if p2 in n else None
    return None

  zeros = list(x).count(0)
  if zeros == 3: return -1
  if zeros == 0:
    return 1 if p2 == 15 else -1 if p2 in {-15, -9, -6, 6, 9} else None

  if zeros == 2:
    if p2 in {12, 16}: return 1
    if p2 == 8:
      p4 = A*C - A*B - B*C
      return 1 if p4 == -4 else -1 if p4 == 0 else None
    if p2 == 5:
      p4_2 = A**2 + B**2 + C**2
      return 1 if p4_2 == 16 else -1 if p4_2 == 4 else None
    if p2 in {-16, -12, 6, 7, 9, 10} or -10 <= p2 <= 4: return -1
    return None

  if zeros == 1:
    p2_2 = sum(v*v for v in x)
    if p2_2 == 2: return -1
    p4 = A*C - A*B - B*C
    if p4 == -2: return -1
    if p2 in {12, 13, 14, 16}: return 1
    if p2 in {-16, 0, 9} or -14 <= p2 <= -2: return -1
    if p4 == 16: return 1
    if p4 in {-16, -8}: return -1
    if p2 in {3, 10}: return 1

    if p4 == 8: return -1 if p2 in {1, 2, 6} else 1 if p2 in {-1, 5, 7} else None
    if p4 == 2: return -1 if p2 in {-1, 1, 2, 4, 6, 8} else 1 if p2 == 5 else None

    if p4 == -4:
      if p2 in {-1, 1, 2, 4, 5, 7}: return -1
      if p2 == 6: return 1
      if p2 in {8, 11}: return 1 if p2_2 == 11 else -1 if p2_2 == 14 else None
      return None

    if p4 == 4:
      if p2 in {-1, 8}: return -1
      if p2 in {2, 11}: return 1
      if p2 in {4, 5, 7}: return -1 if p2_2 == 14 else 1 if p2_2 == 11 else None
      if p2 == 1: return -1 if p2_2 == 14 else None
      return None

  return None
