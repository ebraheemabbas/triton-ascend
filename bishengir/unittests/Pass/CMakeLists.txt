add_bishengir_unittest(BiShengIRPassTests
  FilterPassesTest.cpp
  RegbaseMultiBufferPipelineTest.cpp
)

target_link_libraries(BiShengIRPassTests PRIVATE
  BiShengIRDialectUtils
  BiShengIRHIVMRegbasePipelines
  BiShengIRHIVMPipelines
  BiShengIRMemRefExtDialect
  BiShengIRAnnotationDialect
  MLIRIR
  MLIRPass
  MLIRFuncDialect
)
