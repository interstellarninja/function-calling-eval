import ast
import json
from jsonschema import validate, ValidationError
from pydantic import ValidationError
from utils import eval_logger
from schema import FunctionCall, FunctionSignature

def validate_function_call_schema(call, signatures):
    try:
        call_data = FunctionCall(**call)
    except ValidationError as e:
        eval_logger.info(f"Invalid function call: {e}")
        return False

    for signature in signatures:
        # Inside the main validation function
        try:
            signature_data = FunctionSignature(**signature)
            
            if signature_data.function.name == call_data.name:

                # Validate types in function arguments
                for arg_name, arg_schema in signature_data.function.parameters.get('properties', {}).items():
                    if arg_name in call_data.arguments:
                        call_arg_value = call_data.arguments[arg_name]
                        if call_arg_value:
                            try:
                                validate_argument_type(arg_name, call_arg_value, arg_schema)
                            except Exception as arg_validation_error:
                                eval_logger.info(f"Invalid argument '{arg_name}': {arg_validation_error}")
                                return False

                # Check if all required arguments are present
                required_arguments = signature_data.function.parameters.get('required', [])
                result, missing_arguments = check_required_arguments(call_data.arguments, required_arguments)

                if not result:
                    eval_logger.info(f"Missing required arguments: {missing_arguments}")
                    return False

                return True
        except Exception as e:
            # Handle validation errors for the function signature
            eval_logger.info(f"Error validating function call: {e}")
            return False

    # Moved the "No matching function signature found" message here
    eval_logger.info(f"No matching function signature found for function: {call_data.name}")
    return False

def check_required_arguments(call_arguments, required_arguments):
    missing_arguments = [arg for arg in required_arguments if arg not in call_arguments]
    return not bool(missing_arguments), missing_arguments

def validate_enum_value(arg_name, arg_value, enum_values):
    if arg_value not in enum_values:
        raise Exception(
            f"Invalid value '{arg_value}' for parameter {arg_name}. Expected one of {', '.join(map(str, enum_values))}"
        )

def validate_argument_type(arg_name, arg_value, arg_schema):
    arg_type = arg_schema.get('type', None)
    if arg_type:
        if arg_type == 'string' and 'enum' in arg_schema:
            enum_values = arg_schema['enum']
            if None not in enum_values and enum_values != []:
                try:
                    validate_enum_value(arg_name, arg_value, enum_values)
                except Exception as e:
                    # Propagate the validation error message
                    raise Exception(f"Error validating function call: {e}")

        python_type = get_python_type(arg_type)
        if not isinstance(arg_value, python_type):
            raise Exception(f"Type mismatch for parameter {arg_name}. Expected: {arg_type}, Got: {type(arg_value)}")

def get_python_type(json_type):
    type_mapping = {
        'string': str,
        'number': (int, float),
        'integer': int,
        'boolean': bool,
        'array': list,
        'object': dict,
        'null': type(None),
    }
    return type_mapping[json_type]


def validate_json_data(json_object, json_schema):
    valid = False
    error_message = None
    result_json = None
    
    try:
        # Attempt to load JSON using json.loads
        try:
            result_json = json.loads(json_object)
        except json.decoder.JSONDecodeError:
            # If json.loads fails, try ast.literal_eval
            try:
                result_json = ast.literal_eval(json_object)
            except (SyntaxError, ValueError) as e:
                error_message = f"JSON decoding error: {e}"
                # Return early if both json.loads and ast.literal_eval fail
                eval_logger.info(f"Validation failed for JSON data: {error_message}")
                return valid, result_json

        # Validate each item in the list against schema if it's a list
        if isinstance(result_json, list):
            for index, item in enumerate(result_json):
                try:
                    validate(instance=item, schema=json_schema)
                    eval_logger.info(f"Item {index+1} is valid against the schema.")
                except ValidationError as e:
                    error_message = f"Validation failed for item {index+1}: {e}"
                    break
        else:  # Default to validation without list
            try:
                validate(instance=result_json, schema=json_schema)
                #eval_logger.info("JSON object is valid against the schema.")
            except ValidationError as e:
                error_message = f"Validation failed: {e}"
    except Exception as e:
        error_message = f"Error occurred: {e}"

    if error_message is None:
        valid = True
        eval_logger.info("JSON data is valid against the schema.")
    else:
        eval_logger.info(f"Validation failed for JSON data: {error_message}")

    return valid, result_json


def validate_json_completion(json_obj1, json_obj2):
    # Check if keys match
    try:
        if set(json_obj1.keys()) != set(json_obj2.keys()):
            eval_logger.info("Keys don't match:")
            eval_logger.info(f"Expected: {set(json_obj1.keys())}")
            eval_logger.info(f"Got: {set(json_obj2.keys())}")
            return "failed"

        # Check if values match
        for key in json_obj1.keys():
            if json_obj1[key] != json_obj2[key]:
                eval_logger.info(f"Values don't match for key '{key}'")
                eval_logger.info(f"Expected: {json_obj1[key]}")
                eval_logger.info(f"Got: {json_obj2[key]}")
                return "failed"
    except Exception as e:
        eval_logger.info(f"Exception occured: {e}")
        return "failed"

    # If keys and values match, result remains "passed"
    return "passed"

