/* probe_eval.c — prints eval_classical() for each FEN on stdin, one per line.
 *
 * The C leg of anchor_parity.ps1. It links the LIVE chess-cli engine sources,
 * so what it measures is what the engine actually computes today — not a
 * snapshot that could have drifted. Same reasoning as parity.ps1:10-14, which
 * always compiles a fresh probe against a fresh header rather than trusting a
 * prebuilt one.
 *
 * MUST NOT be built with -DUSE_NNUE. eval.c would then pull in nnue.c and
 * evaluate() would include the net — but the trainer anchors on the bare
 * classical term (tools/trainer.c:34-35, :183).
 *
 * Build (from chess-cli/c_port, mirroring trainer.c:36-38):
 *   zig cc -O2 -std=c11 -mcpu=baseline -I. -o probe_eval.exe \
 *       probe_eval.c bitboard.c position.c eval.c
 *
 * Output: one integer per input line. A FEN the engine REJECTS prints
 * "bad" — the Go side must reject it too, and disagreeing about which FENs are
 * legal is itself a parity failure worth catching.
 */
#include <stdio.h>
#include <string.h>

#include "position.h"
#include "eval.h"

int main(void) {
    position_init();

    char line[512];
    while (fgets(line, sizeof(line), stdin)) {
        size_t n = strlen(line);
        while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
        if (n == 0) continue;

        Position pos;
        if (!pos_from_fen(&pos, line)) {
            printf("bad\n");
        } else {
            printf("%d\n", eval_classical(&pos));
        }
    }
    return 0;
}
