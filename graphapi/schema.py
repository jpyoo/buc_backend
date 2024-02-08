import graphene
from graphene_django import DjangoObjectType
from .models import Todo, Priority

class TodoType(DjangoObjectType):
    class Meta:
        model = Todo
        fields = "__all__"

class PriorityEnum(graphene.Enum):
    LOW = 'LOW'
    NORMAL = 'NORMAL'
    HIGH = 'HIGH'

class CreateTodo(graphene.Mutation):
    class Arguments:
        name = graphene.String(required=True)
        priority = PriorityEnum(required=True)
        completed_at = graphene.DateTime()

    todo = graphene.Field(TodoType)

    def mutate(self, info, name, priority, completed_at=None):
        todo = Todo(name=name, priority=priority, completed_at=completed_at)
        todo.save()
        return CreateTodo(todo=todo)

class UpdateTodo(graphene.Mutation):
    class Arguments:
        id = graphene.ID(required=True)
        name = graphene.String()
        priority = PriorityEnum()
        completed_at = graphene.DateTime()

    todo = graphene.Field(TodoType)

    def mutate(self, info, id, name=None, priority=None, completed_at=None):
        try:
            todo = Todo.objects.get(pk=id)
        except Todo.DoesNotExist:
            raise Exception("Todo not found")

        if name is not None:
            todo.name = name
        if priority is not None:
            todo.priority = priority
        if completed_at is not None:
            todo.completed_at = completed_at

        todo.save()
        return UpdateTodo(todo=todo)

class DeleteTodo(graphene.Mutation):
    class Arguments:
        id = graphene.ID(required=True)

    success = graphene.Boolean()

    def mutate(self, info, id):
        try:
            todo = Todo.objects.get(pk=id)
        except Todo.DoesNotExist:
            raise Exception("Todo not found")

        todo.delete()
        return DeleteTodo(success=True)

class Query(graphene.ObjectType):
    todos = graphene.List(TodoType)

    def resolve_todos(self, info):
        return Todo.objects.all()

class Mutation(graphene.ObjectType):
    create_todo = CreateTodo.Field()
    update_todo = UpdateTodo.Field()
    delete_todo = DeleteTodo.Field()

schema = graphene.Schema(query=Query, mutation=Mutation)