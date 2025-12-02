import config
import serialhelper
import utils
import plot

import argparse
import numpy as np
from datetime import datetime
import time

def main():
    parser = argparse.ArgumentParser(description='GUI tool to visualize the vital signs of multiple persons')

    import os  # 新增的环境变量注入位置

    
    parser.add_argument('userPort', nargs='?', default=os.getenv('USER_PORT', 'COM13'))
    parser.add_argument('dataPort', nargs='?', default=os.getenv('DATA_PORT', 'COM12'))
    parser.add_argument('configFile', nargs='?', default=os.getenv('CFG_FILE', './vod_vs_68xx_10fps.cfg'))
    args = parser.parse_args()

    cfgFile = config.read_config_file(args.configFile)
    cfgFileParsed = config.parse_config_file(cfgFile)
    
    serialUser = serialhelper.SerialHelper(args.userPort, 115200)
    serialData = serialhelper.SerialHelper(args.dataPort, 921600)

    serialUser.sendConfig(cfgFile)
    
    plots = plot.PlotHelper()
    vsData = []

    databuffer = bytearray()
    frameIdx = 0
    extractedValue = np.zeros(20)
    updatedZones = np.zeros((4, 2))

    try:
        # 主循环：持续读取串口数据
        while True:
            newdata = serialData.readall()
            newDebug = serialUser.readall()
            if len(newDebug) != 0:
                print(str(newDebug, 'utf-8'))
            if newdata is None:
                continue
            databuffer += newdata

            magicNumber = False
            magicIdx = databuffer.find(b'\x02\x01\x04\x03\x06\x05\x08\x07')
            if magicIdx != -1:
                databuffer = databuffer[magicIdx:]
            
                totalPacketLength = int.from_bytes(databuffer[8:12], 'little')
                if len(databuffer) >= totalPacketLength:
                    # print('Received packet', frameIdx, totalPacketLength)
                    # frameIdx += 1
                    # databuffer = databuffer[totalPacketLength:]
                    magicNumber = True
            if magicNumber:
                header, index = utils.getHeader(databuffer, 0)
                # print('index header', index)
                frameIdx += 1

                for i in range(header['numTLVs']):
                    tlv, index = utils.getTlv(databuffer, index)
                    # print('tlv', tlv['type'], tlv['length'])
                    # print('index tlv', index)
                    
                    # MMWDEMO_UART_MSG_OD_DEMO_RANGE_AZIMUT_HEAT_MAP（范围-方位热力图）
                    if tlv['type'] == 8:
                        if cfgFileParsed['rangeAzimuthHeatMap'] == 32:
                            rangeAzimuth, index = utils.getOccupDemoRangeAzimuthHeatMap(databuffer, index, cfgFileParsed['numRangeBins'], cfgFileParsed['numAngleBins'])
                            # print('index hm1', index)
                        elif cfgFileParsed['rangeAzimuthHeatMap'] == 16:
                            rangeAzimuth, index = utils.getOccupDemoShortHeatMap(databuffer, index, cfgFileParsed['numRangeBins'], cfgFileParsed['numAngleBins'])
                            # print('index hm2', index, np.min(rangeAzimuth), np.max(rangeAzimuth), np.average(rangeAzimuth))
                        elif cfgFileParsed['rangeAzimuthHeatMap'] == 8:
                            rangeAzimuth, index = utils.getOccupDemoByteHeatMap(databuffer, index, cfgFileParsed['numRangeBins'], cfgFileParsed['numAngleBins'])
                            # print('index hm3', index)
                    # MMWDEMO_UART_MSG_OD_DEMO_DECISION（占用决策）
                    elif tlv['type'] == 9:
                        decisionValue, index = utils.getOccupDemoDecision(databuffer, index, cfgFileParsed['numZones'])
                        # print('{}, {}'.format(decisionValue[0], decisionValue[1]))
                        # print('index des', index)
                    # VS_OUTPUT_HEART_BREATHING_RATES（生命体征输出）
                    elif tlv['type'] == 10:
                        extractedValue, index = utils.getVitalSignsDemoHeartBreathingRate(databuffer, index)
                        # print('P1: HR: {}, RR: {}; P2: HR: {}, RR: {}'.format(np.floor(extractedValue[3]), np.floor(extractedValue[4]), np.floor(extractedValue[8]), np.floor(extractedValue[9])))
                        vsData.append(np.concatenate(([time.time()], extractedValue)))
                    # MMWDEMO_UART_MSG_OD_ROW_NOISE（行噪声数据）
                    elif tlv['type'] == 11:
                        index = utils.dumpRowNoiseValues(databuffer, index, 64)
                        # print('index noise', index)
                    elif tlv['type'] == 12:
                        zoneCount, updatedZones, index = utils.getUpdatedZones(databuffer, index)
                        # print(zoneCount, updatedZones)
                    # MMWDEMO_UART_MSG_STATS（性能统计）
                    elif tlv['type'] == 6:
                        statsInfo, index = utils.getStatsInfo(databuffer, index)
                        print(statsInfo)
                        # Some additional info can be shown with this data
                    else:
                        print('Unprocessed TLV:', tlv['type'])
                plots.update(rangeAzimuth, extractedValue, updatedZones)
                # print('deleting data until', index)
                databuffer = databuffer[index:]
    except KeyboardInterrupt:
        np.savetxt('/home/dirk/thesis/twoPersonVitalSigns/logs/iwr_log/vs_log_' + datetime.now().strftime('%Y-%m-%d#%H:%M:%S') + '.csv', np.array(vsData), delimiter=',', fmt='%.2f')
        

if __name__ == '__main__':
    main()
