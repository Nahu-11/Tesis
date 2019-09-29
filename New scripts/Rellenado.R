cat("\014")
rm(list=ls())
graphics.off()

library(REddyProc)
library(dplyr)

# Hacer un for para todo el rellenado y los graficos

camino <- 'C:\\Users\\nahue\\Desktop\\Tesis_2\\'
setwd(camino)
datos <- fLoadTXTIntoDataframe('Datos\\Para_subir.txt')

datos_con_tiempo <- datos %>% 
  filterLongRuns("NEE") %>% 
  fConvertTimeToPosix('YDH', Year = 'Year', Day = 'DoY', Hour = 'Hour')
Eproc <- sEddyProc$new(
  'Marchi', datos_con_tiempo, c('NEE','Rg','Tair','VPD', 'Ustar'))
Eproc$sSetLocationInfo(LatDeg = 37.703, LongDeg = 57.419, TimeZoneHour = -3)
Eproc$sPlotFingerprint('NEE')
Eproc$sPlotHHFluxes('NEE')
Eproc$sPlotDiurnalCycle('NEE')

Eproc$sEstimateUstarScenarios(
  nSample = 100L, probs = c(0.05, 0.5, 0.95))
Eproc$sMDSGapFillUStarScens('NEE')
grep('NEE_.*_f$', names(Eproc$sExportResults()), value = TRUE)
grep('NEE_.*_fsd$', names(Eproc$sExportResults()), value = TRUE)
Eproc$sPlotFingerprint('NEE_uStar_f')
Eproc$sPlotHHFluxes('NEE_uStar_f')
Eproc$sPlotDiurnalCycle('NEE_uStar_f')

Eproc$sMDSGapFill('Tair', FillAll = FALSE,  minNWarnRunLength = NA)     
Eproc$sMDSGapFill('VPD', FillAll = FALSE,  minNWarnRunLength = NA)
Eproc$sMRFluxPartitionUStarScens()
grep("GPP.*_f$|Reco",names(Eproc$sExportResults()), value = TRUE)
Eproc$sPlotFingerprint('GPP_uStar_f')

FilledEddyData <- Eproc$sExportResults()
uStarSuffixes <- colnames(Eproc$sGetUstarScenarios())[-1]
GPPAggCO2 <- sapply(uStarSuffixes, function(suffix) {
  GPPHalfHour <- FilledEddyData[[paste0("GPP_", suffix, "_f")]]
  mean(GPPHalfHour, na.rm = TRUE)
})
molarMass <- 12.011
GPPAgg <- GPPAggCO2 * 1e-6 * molarMass * 3600*24*365.25
print(GPPAgg)
print((max(GPPAgg)-min(GPPAgg)) / median(GPPAgg))

datos_llenos <- Eproc$sExportResults()
datos_combinados <- cbind(datos, datos_llenos)
fWriteDataframeToFile(datos_combinados, 'Marchi-Results.txt')
