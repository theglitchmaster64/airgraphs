#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib
import os
import io
import sys
import math

def show_plot():
    matplotlib.pyplot.show()
    matplotlib.pyplot.clf()

def save_plot(filename,dpi=1000,replot_existing=False):
    if (not os.path.isfile(filename)):
        matplotlib.pyplot.legend(bbox_to_anchor=(-0.1,1.1))
        matplotlib.pyplot.subplots_adjust(left=0.150)
        matplotlib.pyplot.gcf().set_size_inches(16,9)
        matplotlib.pyplot.savefig(filename,bbox_inches='tight',dpi=dpi)
        matplotlib.pyplot.clf()
        print(filename+' plotted')
        return True
    elif (os.path.isfile(filename) and replot_existing):
        matplotlib.pyplot.legend(bbox_to_anchor=(-0.1,1.1))
        matplotlib.pyplot.subplots_adjust(left=0.150)
        matplotlib.pyplot.gcf().set_size_inches(16,9)
        matplotlib.pyplot.savefig(filename,bbox_inches='tight',dpi=dpi)
        matplotlib.pyplot.clf()
        print(filename+' replotted')
        return True
    else:
        print('plot: {filename} already exists, skipping...\nuse --replot-existing to force new graphs'.format(filename=filename))
        matplotlib.pyplot.clf()
        return False

def label_axes(x=None,y=None):
    if x != None:
        matplotlib.pyplot.xlabel(x)
    if y!= None:
        matplotlib.pyplot.ylabel(y)

def make_mfglookup():
    mfg_txt = open('dataset/mfg_lookup.txt','r').read()
    mfg_txt = mfg_txt.split('\n')
    del mfg_txt[-1]
    mfg_lookup = {}
    for x in mfg_txt:
        mfg_lookup[x.split(':')[0].replace('-','')]=x.split(':')[1]
    return mfg_lookup

def lookup_mfg(key,table):
    try:
        return table[str(key)]
    except:
        return None

def clean_probed(x):
    if type(x) == int:
        return []
    else:
        return x.split(';')

def most_probed_keyset(d,k):
    d[str(k)] = 0


if __name__=='__main__':

    make_plots = False
    replot_existing = False

    if (len(sys.argv) == 1):
        make_plots = False
    elif (sys.argv[1] == '--plot'):
        make_plots = True
        if (len(sys.argv) >= 3):
            if (sys.argv[2] == '--replot-existing'):
                replot_existing = True
    elif (sys.argv[1] == '--test'):
        breakpoint()
    else:
        print('invalid argument!')

    airodump_data = open('dataset/airodump.csv','r').read()

    client_header = 'station-mac, firstseen, lastseen, power, packets, bssid, probed-essids'
    ap_header = 'bssid, firstseen, lastseen, channel, speed, privacy, cipher, authentication, power, beacons, iv, lan-ip, id-length, essid, key'

    client_header_index = airodump_data.index(client_header) #get client header index
    ap_header_index = airodump_data.index(ap_header) #get ap header index

    cl_csv = io.StringIO(airodump_data[client_header_index:])
    ap_csv = io.StringIO(airodump_data[ap_header_index:client_header_index])

    cdf = pd.read_csv(cl_csv, sep=',', skipinitialspace=True, parse_dates=['firstseen','lastseen'], engine='python')
    apdf = pd.read_csv(ap_csv, sep=',', skipinitialspace=True, parse_dates=['firstseen','lastseen'], engine='python')

    #clean client probed-essids column
    cdf['probed-essids'] = cdf['probed-essids'].fillna(0).apply(clean_probed)
    apdf['essid'].fillna('Unknown/Hidden SSID',inplace=True)


    #AP analysis

    #how many ssids
    ssid_stats = apdf['essid'].value_counts(dropna=False)
    if make_plots:
        ssid_stats.plot(kind='bar')
        label_axes(x='ssid',y='no.of access points')
        save_plot('graphs/ssid_stats.png',dpi=500,replot_existing=replot_existing)

    #most occupied channels
    channel_occupied = apdf['channel'].value_counts()
    if make_plots:
        channel_occupied.plot(kind='bar')
        label_axes(x='channel',y='no. of access points')
        save_plot('graphs/channels_occupied.png',dpi=500,replot_existing=replot_existing)

    #what does the channel-essid distribution look like
    channel_essid_distrib = apdf.groupby(['channel','essid'])['channel'].count().unstack().fillna(0).T
    if make_plots:
        channel_essid_distrib.plot(kind='bar',stacked=False)
        label_axes(y='no of access points')
        save_plot('graphs/channel_ssid_distribution.png',dpi=500,replot_existing=replot_existing)

    #most popular manufacturers
    mfg_table = make_mfglookup()
    mfg_stats = apdf['bssid'].apply(lambda x: lookup_mfg(x.replace(':','')[0:6], table=mfg_table))
    if make_plots:
        mfg_stats.fillna('Unknown').value_counts().plot(kind='barh')
        label_axes(y='no. of access points',x='manufacturer')
        save_plot('graphs/manufacturer_by_popularity.png',dpi=500,replot_existing=replot_existing)

    #privacy statistics
    privacy_stats = apdf['privacy'].fillna('Unknown').value_counts()
    if make_plots:
        privacy_stats.plot(kind='bar')
        label_axes(x='standard',y='no of access points')
        save_plot('graphs/privacy_stats.png',dpi=500,replot_existing=replot_existing)

    #which ssids use which privacy
    privacy_ssid = apdf.groupby(['essid','privacy'])['privacy'].count().unstack()
    if make_plots:
        privacy_ssid.plot(kind='bar',stacked=True)
        label_axes(x='ssid',y='no of access points')
        save_plot('graphs/ssid_privacy.png',dpi=500,replot_existing=replot_existing)

    #closest AP from capture device
    closest_aps_df = apdf[apdf['power'] != -1] #if power is unknown, it is set to -1; filter out the -1 power APs
    closest_aps_df = closest_aps_df[['bssid','power']].sort_values('power',ascending = False).head(20).iloc[::-1]
    if make_plots:
        closest_aps_df.plot(kind='barh',x='bssid')
        label_axes(y='bssid',x='power in dB')
        save_plot('graphs/closest_aps.png',dpi=500,replot_existing=replot_existing)
        matplotlib.pyplot.clf()
        closest_aps_df.insert(2,'base10',closest_aps_df['power'].apply(lambda x: math.pow(10,x/20)))
        closest_aps_df.drop(labels='power',axis=1,inplace=True)
        closest_aps_df.plot(kind='barh',x='bssid')
        label_axes(y='bssid',x='power in base10')
        save_plot('graphs/closest_aps_base10.png',dpi=500,replot_existing=replot_existing)

    #discovery timeline
    discovery_timeline_df = apdf[['bssid','firstseen']]
    discovery_timeline_df = discovery_timeline_df.sort_values('firstseen')
    epoch = discovery_timeline_df['firstseen'].iloc[0]
    discovery_timeline_df.insert(2,'time_delta',discovery_timeline_df['firstseen'].apply(lambda x: (x-epoch)/np.timedelta64(1,'s') ) )
    if make_plots:
        discovery_timeline_df.groupby('time_delta').count().plot()
        label_axes(x='time since scan initiatied',y='number of APs discovered')
        save_plot('graphs/discovery_timeline.png',dpi=500,replot_existing=replot_existing)

    #client analysis


    #which client probed most ssids
    probed_df = cdf[['station-mac','probed-essids']].agg({'station-mac': lambda x: x, 'probed-essids': lambda x: len(x)})
    if make_plots:
        probed_df = probed_df.sort_values(by='probed-essids',ascending=False)
        probed_df.head(25).plot(kind='bar',x='station-mac',y='probed-essids')
        label_axes(x='client MAC',y='number of APs probed')
        save_plot('graphs/client_probed.png',dpi=750,replot_existing=replot_existing)

    #most probed ssids
    most_probed_df = cdf[['station-mac','probed-essids']]
    most_probed_stats = {}
    for i in most_probed_df['probed-essids']:
        for j in i:
            try:
                most_probed_stats[str(j)] += 1
            except KeyError:
                most_probed_keyset(most_probed_stats,j)
                most_probed_stats[str(j)] += 1
    del most_probed_df
    most_probed_df = pd.DataFrame(most_probed_stats,index=[0]).T
    if make_plots:
        most_probed_df.sort_values(by=0,ascending=False,inplace=True)
        most_probed_df.head(25).plot(kind='bar')
        label_axes(x='AP ssid',y='number of times probed')
        save_plot('graphs/most_probed_AP_ssid.png',dpi=500,replot_existing=replot_existing)

    #what % are roaming vs connected
    cvr_pie_df = cdf['bssid'].str.contains('not associated').value_counts().rename({True:'roaming',False:'connected'})
    roaming = float(cvr_pie_df['roaming'])
    connected = float(cvr_pie_df['connected'])
    total = roaming+connected
    roaming_perc = (roaming/total)*100
    connected_perc = (connected/total)*100
    if make_plots:
        cvr_pie_df.plot(kind='pie')
        label_axes(y='',x='roaming:{}%\nconnected:{}'.format(roaming_perc,connected_perc))
        save_plot('graphs/roaming_vs_connected.png',dpi=500,replot_existing=replot_existing)

    #client v AP distribution
    connected_clients = (cdf[~cdf['bssid'].str.contains('not associated')].reset_index(drop=True))[['station-mac','bssid']]
    available_networks = apdf[['bssid','essid']]
    client_network_map = pd.merge(connected_clients,available_networks,on='bssid')
    distribution_stats = client_network_map.groupby('essid')['station-mac'].count()
    if make_plots:
        distribution_stats.plot(kind='bar')
        save_plot('graphs/client_AP_distribution.png',dpi=500,replot_existing=replot_existing)
