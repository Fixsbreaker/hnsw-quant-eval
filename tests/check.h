#pragma once

#include <cstdio>
#include <cstdlib>

// Minimal test helpers, no external framework needed.
#define CHECK(cond)                                                           \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__,       \
                         __LINE__, #cond);                                    \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

#define CHECK_THROWS(expr)                                                    \
    do {                                                                      \
        bool thrown = false;                                                  \
        try {                                                                 \
            expr;                                                             \
        } catch (...) {                                                       \
            thrown = true;                                                    \
        }                                                                     \
        CHECK(thrown);                                                        \
    } while (0)
