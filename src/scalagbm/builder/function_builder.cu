
#include <scalagbm/builder/function_builder.h>
#include "scalagbm/builder/exact_tree_builder.h"
#include "scalagbm/builder/hist_tree_builder.h"

FunctionBuilder *FunctionBuilder::create(std::string name) {
    if (name == "exact") return new ExactTreeBuilder;
    if (name == "hist") return new HistTreeBuilder;
    LOG(FATAL) << "unknown builder " << name;
    return nullptr;
}

