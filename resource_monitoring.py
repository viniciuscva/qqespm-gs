
import psutil
import pandas as pd
import subprocess
import qqespm_sql_exp as qqsql3

totalmem = psutil.virtual_memory().total/(1024*1024)
shared_buffer_size = qqsql3.get_shared_buffers_size()

# def get_shared_buffers_size(config_filename = 'config/general_connector.ini'):
#     sql = """select cast(setting as numeric) * 8192/(1024*1024) as shared_buffer_size from  
#     pg_settings where name='shared_buffers';"""

#     conn = qqsql3.establish_postgis_connection(config_filename=config_filename)
#     cur = conn.cursor()
#     cur.execute(sql)
#     shared_buffer_size = float(cur.fetchall()[0][0])
#     cur.close()
#     conn.close()
#     return shared_buffer_size

def getProcessesInfo(username = None, command_keyword = None, as_dataframe = True):
    '''
    Get list of running process sorted by Memory Usage
    '''
    listOfProcObjects = []
    # Iterate over the list
    for proc in psutil.process_iter():
        try:
            # Fetch process details as dict
            pinfo = proc.as_dict(attrs=['pid', 'name', 'username', 'cmdline', 'cpu_percent', 'num_threads', 'memory_percent', 'cpu_num', 'status', 'create_time'])
            pinfo['cmdline'] = ' '.join(pinfo['cmdline'])
            pinfo['vms'] = proc.memory_info().vms / (1024 * 1024)
            pinfo['rss'] = proc.memory_info().rss / (1024 * 1024)
            pinfo['shared'] = proc.memory_info().shared / (1024 * 1024)

            # Append dict to list
            if username is not None and command_keyword is not None:
                if username == pinfo['username'] and command_keyword in pinfo['cmdline']:
                    listOfProcObjects.append(pinfo)
            elif username is not None:
                if username == pinfo['username']:
                    listOfProcObjects.append(pinfo)
            elif command_keyword is not None:
                if command_keyword in pinfo['cmdline']:#any([command_keyword in e for e in pinfo['cmdline']]):
                    listOfProcObjects.append(pinfo)
            else:
                listOfProcObjects.append(pinfo)
            

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    # Sort list of dict by key vms i.e. memory usage
    listOfProcObjects = sorted(listOfProcObjects, key=lambda procObj: procObj['rss'], reverse=True)
    if as_dataframe is False:
        return listOfProcObjects
    processes_df = pd.DataFrame(listOfProcObjects)
    processes_df = processes_df[['pid','username','name','cmdline','cpu_percent','memory_percent','rss','vms','shared','num_threads','cpu_num','status','create_time']]

    return processes_df

def get_total_resource_usage_info(username = None, command_keyword = None):
    processes_info = getProcessesInfo(username, command_keyword)[['memory_percent','rss','vms','shared']]
    #resource_usage_info = processes_info.sum(axis=0).to_dict()
    resource_usage_info = {'max_shared': processes_info['shared'].max()}
    resource_usage_info['unshared_memory'] = processes_info['rss'].sum() - processes_info['shared'].sum()
    return resource_usage_info

def get_total_memory_usage_for_elastic():
    resource_usage_elastic = get_total_resource_usage_info(username = 'elasticsearch')
    total_memory_usage_elastic = resource_usage_elastic['unshared_memory'] + resource_usage_elastic['max_shared']
    return total_memory_usage_elastic

def get_total_memory_usage_for_postgres():
    global shared_buffer_size

    resource_usage_postgres = get_total_resource_usage_info(username = 'postgres')
    total_memory_usage_postgres = resource_usage_postgres['unshared_memory'] + shared_buffer_size
    return total_memory_usage_postgres

def get_total_memory_usage_for_qqespm():
    try:
        total_memory_usage_qqespm = getProcessesInfo(command_keyword='compare_modules_experiments')['rss'].max()
    except KeyError:
        total_memory_usage_qqespm = 0
    return total_memory_usage_qqespm

#print(getProcessesInfo(username='elasticsearch').shape)
#print(getProcessesInfoWithPS(username='elasticsearch').shape)
#getProcessesInfo(username='elasticsearch') # -> Dataframe with processes resource use information


def getProcessesInfoWithPS(username = None):
    ps = subprocess.Popen(['ps', '-e', 'aux'], stdout=subprocess.PIPE).communicate()[0]
    processes = ps.decode('utf8').split('\n')
    for i, process in enumerate(processes):
        processes[i] = process.split()
    column_names = processes[0]

    df = pd.DataFrame(processes[1:])
    df.fillna('', inplace=True)
    df.loc[:, 11] = df.loc[:, 11:].sum(axis=1)
    df.rename({i: column_names[i] for i in range(11)}, axis=1, inplace=True)
    df = df[column_names]
    if username is not None:
        df = df[df['USER'] == username]
    df.loc[df['%CPU']=='', '%CPU'] = None
    df.loc[df['%MEM']=='', '%MEM'] = None
    df.loc[df['VSZ']=='', 'VSZ'] = None
    df.loc[df['RSS']=='', 'RSS'] = None
    df['%CPU'] = df['%CPU'].astype(float)
    df['%MEM'] = df['%MEM'].astype(float)
    df['VSZ'] = df['VSZ'].astype(float)
    df['RSS'] = df['RSS'].astype(float)
    return df

# getProcessesInfoWithPS().sort_values(by='%MEM', ascending=False)[['%CPU','%MEM','VSZ','RSS']].sum()
