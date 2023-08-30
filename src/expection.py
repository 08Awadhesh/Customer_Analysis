import sys
from src.logger import logging

def error_message_detail(error,error_detial:sys):
        _,_,exc_tb=error_detial.exc_info()
        file_name=exc_tb.tb_frame.f_code.co_filename

        errr_message = "Error occured in python script name [{0}] line number [{1}] error message" 
        file_name, exc_tb.tb_lineno, str(error)

        return errr_message  


class CustomException(Exception):

    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detial(error_message,error_detail=error_detail)


    def __str__(self):
        return self.error_message


        