from abc import abstractmethod
import psycopg2 

class DB():
    def __init__(self, config):
        self.config = config 

    @abstractmethod
    def connect():
        pass 

class DBConnection(DB):
    def __init__(self, config):
        super().__init__(config)
    
    def connect(self):
        self.conn = psycopg2.connect(
            host=self.config['host'],
            dbname=self.config['db_name'],
            user=self.config['user_id'],
            password=self.config['user_pw'],
            port=self.config['port']
        )
        self.cur = self.conn.cursor()

    def close(self):
        self.cur.close()
        self.conn.close()

class PostgresDB:
    '''
    데이터베이스 CRUD (Create, Read, Update, Delete) 작업에 사용되는 클래스
    '''
    def __init__(self, db_connection):
        self.db_connection = db_connection

    def get_total_data(self, table_name):
        '''
        테이블에서 전체 데이터를 가져옵니다. 
        args:
        table_name (str)

        returns:
        list[str]  - 자료형 확인 필요: 테이블 전체 데이터 
        '''
        query = f"SELECT * FROM {table_name};"
        self.db_connection.cur.execute(query)
        return self.db_connection.cur.fetchall()

    def get_conf_log_data(self, meeting_id):
        query = f"SELECT * FROM ibk_poc_conf_log WHERE conf_id = {meeting_id};"
        self.db_connection.cur.execute(query)
        return self.db_connection.cur.fetchall()

    def get_day_data(self, table_name, date):
        '''
        테이블에서 전일 데이터를 가져옵니다.
        args:
        date (str): 20240130 형식
        '''
        query = f"SELECT * FROM {table_name} WHERE SPLIT_PART(conv_id, '_', 1) = '{date}';"
        self.db_connection.cur.execute(query)
        return self.db_connection.cur.fetchall()

    def get_null_data(self):
        query = f"SELECT conf_id FROM ibk_poc_conf_log GROUP BY conf_id HAVING COUNT(conv_id) = 0;"
        self.db_connection.cur.execute(query)
        return self.db_connection.cur.fetchall()

    def get_mismatch_data(self):
        query = f"SELECT conf_id FROM ibk_poc_conf WHERE conf_id NOT IN (SELECT conf_id FROM ibk_poc_conf_log)"
        self.db_connection.cur.execute(query)
        data_ids = self.db_connection.cur.fetchall()
        data_list = [data_id[0] for data_id in data_ids]
        return data_list

    def check_pk(self, table_name, pk_value):
        '''
        테이블에 Primary Key(PK)가 존재하는지 확인합니다. 이미 존재하는 PK인 경우, True를 반환합니다. 
        args:
        data (list): 테이블 행 데이터   ex) [conv_id, date, qa, content, user_id]
        '''
        self.db_connection.cur.execute(f"SELECT EXISTS(SELECT 1 FROM {table_name} WHERE conv_id = %s)", (pk_value,))
        return self.db_connection.cur.fetchone()[0]  # True 또는 False 반환


class TableEditor:
    def __init__(self, db_connection):
        self.db_connection = db_connection

    def edit_user_tb(self, task, table_name, data_type=None, data=None, col=None, val=None):
        '''
        insert, delete, update
        data_type = raw or table
        '''
        if task == 'insert':
            if data_type == 'table':
                for idx in range(len(data)):
                    self.db_connection.cur.execute(
                        f"INSERT INTO {table_name} (user_id, is_active, name, company) VALUES (%s, %s, %s, %s)",
                        (data['user_id'][idx], data['is_active'][idx])
                    )
                    self.db_connection.conn.commit()
            elif data_type == 'raw':
                self.db_connection.cur.execute(
                    f"INSERT INTO {table_name} (user_id, is_active, name, company) VALUES (%s, %s, %s, %s)",
                    tuple(data)
                )
                self.db_connection.conn.commit()
        elif task == 'delete':
            self.db_connection.cur.execute(
                    f"""DELETE FROM {table_name} WHERE conf_id = %s""",
                    (data, )
            )
            self.db_connection.conn.commit()
        elif task == 'update':
            pass

    def edit_poc_conf_tb(self, task, table_name, data=None):
        if task == 'insert':
            pass
        elif task == 'delete':
            self.db_connection.cur.execute(
                f"""DELETE FROM {table_name} WHERE conf_id = %s""",
                (data, )
            )
            self.db_connection.conn.commit()
        elif task == 'update':   # data  - conf_id 
            self.db_connection.cur.execute(
                    f"""
                    UPDATE {table_name}
                    SET stt_sign = true
                    WHERE conf_id = %s""",
                    (data, )
            )
            self.db_connection.conn.commit()

    def get_unknown_id(self, conf_id, val='UNKNOWN'):
        self.db_connection.cur.execute(
            f"""SELECT cuser_id FROM ibk_poc_conf_user WHERE conf_id=%s and speaker_id=%s""",
            (conf_id, val)
        )
        cuser_result = self.db_connection.cur.fetchone()
        return cuser_result
    
    def edit_poc_conf_log_tb(self, task, table_name, data=None, val=None):     
        if task == 'insert':    # data - meeting_id, val: (start time, end time, content)
            self.db_connection.cur.execute(
                f"""SELECT cuser_id FROM ibk_poc_conf_user WHERE conf_id=%s and speaker_id=%s""",
                (data, val[3])
            )
            cuser_result = self.db_connection.cur.fetchone()
            self.db_connection.cur.execute(
               f"""INSERT INTO {table_name} (start_time, end_time, content, cuser_id, conf_id) VALUES (%s, %s, %s, %s, %s)""",
               (val[0], val[1], val[2], cuser_result, data)
            )
            self.db_connection.conn.commit()
        elif task == 'delete':
            self.db_connection.cur.execute(
                f"""DELETE FROM {table_name} WHERE conf_id = %s""",
                (data, )
            )
            self.db_connection.conn.commit()
        elif task == 'update':   # data: meeting_id, val: (conv_id, speaker_id - SPEAKER_00)
            self.db_connection.cur.execute(
                f"""SELECT cuser_id FROM ibk_poc_conf_user WHERE conf_id=%s and speaker_id=%s""",
                (data, val[1])
            )
            cuser_result = self.db_connection.cur.fetchone()
            # print(f'speaker_id: {speaker_id}, cuser: {cuser_result}')
            self.db_connection.cur.execute(
                f"""UPDATE {table_name} 
                    SET cuser_id = %s
                    WHERE conf_id = %s AND conv_id = %s""",
                    (cuser_result, data, val[0])
            )
            self.db_connection.conn.commit()

    def bulk_insert(self, table_name, columns, values):
        print(f'table_name: {table_name}, columns: {columns}, values: {values}')
        columns_str = ', '.join(columns)
        placeholders = ', '.join(['%s'] * len(columns))
        query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
        
        self.db_connection.cur.executemany(query, values)
        self.db_connection.conn.commit()
