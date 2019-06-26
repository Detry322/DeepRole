TARGET := deeprole

CC := clang++

EXTRA_CFLAGS ?=
CFLAGS := -Wall -std=c++17 -Ilibs/

ifeq ($(DEBUG),1)
  CFLAGS += -O0 -g
else ifeq ($(PROFILE), 1)
  CFLAGS += -O1 -g -DNDEBUG
else
  CFLAGS += -O3 -DNDEBUG
endif

CFLAGS += $(EXTRA_CFLAGS)

LDFLAGS :=

ifeq ($(OPENMIND),1)
  CC := clang++-8
  CFLAGS += -march=corei7-avx -static -mavx -DOPENMIND -I/usr/include/c++/5/ -I/usr/include/x86_64-linux-gnu/c++/5/ -nostdinc++
  LDFLAGS += -march=corei7-avx -static -mavx -DOPENMIND -I/usr/include/c++/5/ -I/usr/include/x86_64-linux-gnu/c++/5/ -nostdinc++
endif

# You shouldn't need to touch this.  This keeps track of whether or
# not you've changed CFLAGS.
OLD_CFLAGS := $(shell cat .cflags 2> /dev/null)
ifneq ($(CFLAGS),$(OLD_CFLAGS))
.cflags::
	@echo "$(CFLAGS)" > .cflags
endif


all: $(TARGET)

%.o: %.cpp .cflags
	$(CC) $(CFLAGS) -c $<

$(TARGET): main.o lookahead.o lookup_tables.o cfr_plus.o util.o nn.o serialization.o fdeep_replace.o
	$(CC) -o $(TARGET) $^ $(LDFLAGS)

clean::
	rm -f $(TARGET) *.o *.s .cflags
