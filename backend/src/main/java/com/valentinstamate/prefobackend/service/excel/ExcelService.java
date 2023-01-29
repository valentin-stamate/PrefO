package com.valentinstamate.prefobackend.service.excel;

import jakarta.enterprise.context.ApplicationScoped;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.util.*;

public class ExcelService {

    public static Map<Integer, List<Object>> readSheet(Workbook workbook, int sheetNumber) {
        Sheet sheet = workbook.getSheetAt(sheetNumber);

        Map<Integer, List<Object>> data = new HashMap<>();

        int rowCount = 0;

        for (Row row : sheet) {
            List<Object> rowList = new ArrayList<>();
            data.put(rowCount, rowList);

            for (Cell cell : row) {
                switch (cell.getCellType()) {
                    case STRING:
                        rowList.add(cell.getStringCellValue());
                        break;
                    case NUMERIC:
                        if (DateUtil.isCellDateFormatted(cell)) {
                            rowList.add(cell.getDateCellValue());
                        } else {
                            rowList.add(cell.getNumericCellValue());
                        }
                        break;
                    case BOOLEAN:
                        rowList.add(cell.getBooleanCellValue());
                        break;
                    case FORMULA:
                        rowList.add(cell.getCellFormula());
                        break;
                    default:
                        rowList.add("");
                        break;
                }
            }

            rowCount++;
        }

        return data;
    }

    public static Workbook createSheet(Map<Integer, List<Object>> data, String sheetName) {
        Workbook workbook = new XSSFWorkbook();

        Sheet sheet = workbook.createSheet(sheetName);

        for (var rowEntry : data.entrySet()) {
            int rowCount = rowEntry.getKey();
            List<Object> rowValues = rowEntry.getValue();

            Row sheetRow = sheet.createRow(rowCount);

            for (int cellCount = 0; cellCount < rowValues.size(); cellCount++) {
                Cell cell = sheetRow.createCell(cellCount);
                Object cellValue = rowValues.get(cellCount);

                if (cellValue instanceof Double) {
                    cell.setCellValue((Double) cellValue);
                } else if (cellValue instanceof Date) {
                    CellStyle cellStyle = workbook.createCellStyle();
                    CreationHelper creationHelper = workbook.getCreationHelper();
                    cellStyle.setDataFormat(creationHelper.createDataFormat().getFormat("d/m/yy h.mm;@"));

                    cell.setCellValue((Date) cellValue);
                    cell.setCellStyle(cellStyle);
                } else {
                    cell.setCellValue("" + cellValue);
                }

            }

        }

        return workbook;
    }

    public static int parseNumericCol(Object value) {
        if (value instanceof Integer) {
            return (Integer) value;
        }

        if (value instanceof Double) {
            return ((Double) value).intValue();
        }

        if (value instanceof String) {
            try {
                return Integer.parseInt((String) value);
            } catch (Exception e) {
                return 0;
            }
        }

        return 0;
    }

    public static Map<Integer, List<Object>> filterInvalidRows(Map<Integer, List<Object>> data) {
        var filteredData = new HashMap<Integer, List<Object>>();

        var row = 1;
        for (var entry : data.entrySet()) {
            var key = entry.getKey();
            var value = entry.getValue();

            var found = false;

            for (var e : value) {
                if (e.equals("")) {
                    found = true;
                    break;
                }
            }

            if (!found) {
                filteredData.put(row++, value);
            }
        }

        return filteredData;
    }

}
