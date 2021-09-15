# Copyright (C) 2017 ETH Zurich, University of Bologna and GreenWaves Technologies
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

PULP_APP = cnnOps
PULP_APP_FC_SRCS = main.c
PULP_CFLAGS = -O2 -g

#RISCV STD ISA
#RISCV_FLAGS = -march=rv32imc -DRISCV

#RISCV STD ISA + GAP8 Extensions
RISCV_FLAGS ?= -march=rv32imcxgap8


RISCV_FLAGS += -mPE=8 -mFC=1 -D__riscv__

# Please incude the "pulp_rules.mk" according to your path
include /gap8/gap_sdk/tools/rules/pulp_rules.mk